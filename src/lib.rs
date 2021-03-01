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

#[derive(Debug, Clone, PartialEq)]
pub enum Term<V, T> {
    Variable(V),
    Constant(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom<V, S, T> {
    pub symbol: S,
    pub terms: Vec<Term<V, T>>,
}

impl<V: Clone, S, T> Atom<V, S, T> {
    pub fn new(symbol: S, terms: Vec<Term<V, T>>) -> Self {
        Self { symbol, terms }
    }

    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(v.clone()),
            Term::Constant(_) => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Query<V, S, T> {
    pub vars: Vec<V>,
    pub atoms: Vec<Atom<V, S, T>>,
}

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone,
{
    pub fn new(atoms: Vec<Atom<V, S, T>>) -> Self {
        let mut new = Self {
            atoms,
            vars: vec![],
        };

        for atom in &new.atoms {
            for var in atom.vars() {
                if new.index_of(&var).is_none() {
                    new.vars.push(var)
                }
            }
        }
        new
    }

    pub fn index_of(&self, var: &V) -> Option<usize> {
        self.vars.iter().position(|v| v == var)
    }

    fn split(mut self) -> (Self, Self) {
        assert!(self.atoms.len() >= 2);
        let other_atoms = self.atoms.split_off(self.atoms.len() / 2);
        (Self::new(self.atoms), Self::new(other_atoms))
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
    T: Data + 'static,
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

                let mut used_vars: FxHashMap<V, Vec<usize>> = Default::default();
                let mut term_eqs: Vec<(T, usize)> = vec![];
                for (i, term) in atom.terms.iter().enumerate() {
                    match term {
                        Term::Variable(v) => used_vars.entry(v.clone()).or_default().push(i),
                        Term::Constant(c) => term_eqs.push((c.clone(), i)),
                    }
                }

                let mut var_eqs: Vec<(usize, usize)> = vec![];
                for (_v, occurences) in used_vars {
                    for pair in occurences.windows(2) {
                        var_eqs.push((pair[0], pair[1]))
                    }
                }

                Scan {
                    relation: atom.symbol,
                    var_eqs,
                    term_eqs,
                }
                .into_dyn()
            }
            _ => {
                let my_vars = self.vars.clone();
                let (q1, q2) = self.split();
                let (key1, key2): (Vec<usize>, Vec<usize>) = q1
                    .vars
                    .iter()
                    .enumerate()
                    .filter_map(|(i1, v)| q2.index_of(v).map(|i2| (i1, i2)))
                    .unzip();

                let merge: Vec<Sided<usize>> = my_vars
                    .into_iter()
                    .map(|v| {
                        if let Some(i) = q1.index_of(&v) {
                            Sided::Left(i)
                        } else if let Some(i) = q2.index_of(&v) {
                            Sided::Right(i)
                        } else {
                            unreachable!("var has to come from one side")
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
        assert_eq!(q.vars, vec![0]);
    }
}
