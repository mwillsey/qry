use fxhash::FxHashMap;
use std::{fmt::Debug, hash::Hash, slice::ChunksExact};

mod expr;
pub use expr::*;

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq + Default {}
impl<T> Data for T where T: Debug + Clone + Hash + PartialOrd + Eq + Default {}

pub trait Database {
    type S: RelationSymbol;
    type T: Data;
    fn get(&self, sym: &Self::S) -> ChunksExact<Self::T>;
}

#[derive(Default)]
struct SimpleDatabase<S, T> {
    pub map: FxHashMap<S, (usize, Vec<T>)>,
}

impl<S, T> Database for SimpleDatabase<S, T>
where
    S: RelationSymbol,
    T: Data,
{
    type S = S;
    type T = T;

    fn get(&self, sym: &Self::S) -> ChunksExact<Self::T> {
        let (arity, ts) = &self.map[sym];
        ts.chunks_exact(*arity)
    }
}

pub type Var = u32;

#[derive(Debug, Clone, PartialEq)]
pub enum Term<T> {
    Variable(Var),
    Constant(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom<S, T> {
    pub symbol: S,
    pub terms: Vec<Term<T>>,
}

impl<S, T> Atom<S, T> {
    fn vars(&self) -> impl Iterator<Item = Var> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(*v),
            Term::Constant(_) => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Query<S, T> {
    n_vars: usize,
    atoms: Vec<Atom<S, T>>,
}

type VarMap = FxHashMap<Var, usize>;

impl<S, T> Query<S, T> {
    pub fn new(atoms: Vec<Atom<S, T>>) -> Self {
        let mut vars = vec![];
        for atom in &atoms {
            for term in &atom.terms {
                if let Term::Variable(v) = term {
                    vars.push(*v);
                }
            }
        }
        vars.sort_unstable();
        vars.dedup();
        let n_vars = vars.len();
        assert!(vars.iter().map(|i| *i as usize).eq(0..n_vars));
        Self { atoms, n_vars }
    }

    pub fn relabel(mut atoms: Vec<Atom<S, T>>) -> (Self, VarMap) {
        let mut vars = VarMap::default();
        for atom in &mut atoms {
            for term in &mut atom.terms {
                if let Term::Variable(ref mut v) = term {
                    let n = vars.len();
                    *v = *vars.entry(*v).or_insert(n) as Var;
                }
            }
        }
        let n_vars = vars.len();
        (Self { atoms, n_vars }, vars)
    }

    fn split(mut self) -> (Self, VarMap, Self, VarMap) {
        assert!(self.atoms.len() >= 2);

        let other_atoms = self.atoms.split_off(self.atoms.len() / 2);

        let (left, lv) = Self::relabel(self.atoms);
        let (right, rv) = Self::relabel(other_atoms);

        (left, lv, right, rv)
    }
}

impl<S, T> Query<S, T>
where
    S: RelationSymbol + 'static,
    T: Data,
{
    pub fn compile<DB>(self) -> DynExpression<DB>
    where
        DB: Database<S = S, T = T> + 'static,
    {
        match self.atoms.len() {
            0 => panic!(),
            1 => {
                let mut atoms = self.atoms;
                let atom = atoms.pop().unwrap();

                let vars: Vec<Var> = atom.vars().collect();
                let n_vars = vars.len();
                if n_vars == atom.terms.len() && vars.into_iter().eq(0..n_vars as Var) {
                    Scan::new(atom.symbol).into_dyn()
                } else {
                    // if there is non-linearity or constants, we just can't handle it rn
                    unimplemented!()
                }
            }
            _ => {
                let n_vars = self.n_vars;
                let (q1, v1, q2, v2) = self.split();
                let (key1, key2): (Vec<usize>, Vec<usize>) = v1
                    .iter()
                    .filter_map(|(v, i1)| v2.get(v).map(|i2| (i1, i2)))
                    .unzip();

                let merge: Vec<Sided<usize>> = (0..n_vars as Var)
                    .map(|v| {
                        if let Some(i) = v1.get(&v) {
                            Sided::Left(*i)
                        } else if let Some(i) = v2.get(&v) {
                            Sided::Right(*i)
                        } else {
                            panic!()
                        }
                    })
                    .collect();

                HashJoin {
                    key1,
                    key2,
                    merge,
                    expr1: q1.compile(),
                    expr2: q2.compile(),
                }
                .into_dyn()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! query {
        ($( $sym:ident ( $($term:expr),+ ) ),+) => {
            Query::new(vec![$(
                Atom {
                    symbol: stringify!($sym),
                    terms: vec![$(query!(@term $term)),*],
                }
            ),+])
        };
        (@term $e:literal ) => { Term::Variable($e) };
        (@term $e:expr ) => { Term::Constant($e) };
    }

    #[test]
    fn query_macro() {
        let q = query!(r(0, { "foo" }));
        let q_expected = Query::new(vec![Atom {
            symbol: "r",
            terms: vec![Term::Variable(0), Term::Constant("foo")],
        }]);
        assert_eq!(q, q_expected);
        assert_eq!(q.n_vars, 1);
    }
}
