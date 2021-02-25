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
pub enum Term<T> {
    Variable(u32),
    Constant(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom<S, T> {
    pub symbol: S,
    pub terms: Vec<Term<T>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Query<S, T> {
    n_vars: usize,
    atoms: Vec<Atom<S, T>>,
}

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
