use std::{
    borrow::Borrow,
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    hash::Hash,
    ops::{Index, IndexMut},
    slice::ChunksExact,
};

mod expr;

pub use expr::*;

#[derive(Debug, Clone, Default)]
pub struct Database<S, T> {
    relations: HashMap<S, Relation<T>>,
}

impl<S: RelationSymbol, T: Data> Database<S, T> {
    pub fn add_relation(&mut self, symbol: S, arity: usize) -> &mut Relation<T> {
        match self.relations.entry(symbol) {
            Entry::Occupied(e) => panic!("Relation {:?} already present", e.key()),
            Entry::Vacant(e) => e.insert(Relation::new(arity)),
        }
    }
}

impl<K, S, T> Index<K> for Database<S, T>
where
    K: Borrow<S>,
    S: RelationSymbol,
    T: Data,
{
    type Output = Relation<T>;
    fn index(&self, symbol: K) -> &Self::Output {
        self.relations.get(symbol.borrow()).unwrap()
    }
}

impl<K, S, T> IndexMut<K> for Database<S, T>
where
    K: Borrow<S>,
    S: RelationSymbol,
    T: Data,
{
    fn index_mut(&mut self, symbol: K) -> &mut Self::Output {
        self.relations.get_mut(symbol.borrow()).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Relation<T> {
    arity: usize,
    ts: Vec<T>,
}

impl<T> Relation<T> {
    pub fn new(arity: usize) -> Self {
        let ts = Default::default();
        Self { arity, ts }
    }

    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.ts.len() % self.arity, 0);
        self.ts.len() / self.arity
    }

    pub fn insert(&mut self, tuple: &[T]) -> &mut Self
    where
        T: Clone,
    {
        assert_eq!(self.arity, tuple.len());
        self.ts.extend_from_slice(tuple);
        self
    }

    pub fn iter(&self) -> ChunksExact<T> {
        self.ts.chunks_exact(self.arity)
    }
}

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq + Default {}
impl<T> Data for T where T: Debug + Clone + Hash + PartialOrd + Eq + Default {}

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
        let mut db = Database::default();
        db.add_relation("r", 2).insert(&[1, 1]);

        let q = query!(r(0, { "foo" }));
        let q_expected = Query::new(vec![Atom {
            symbol: "r",
            terms: vec![Term::Variable(0), Term::Constant("foo")],
        }]);
        assert_eq!(q, q_expected);
        assert_eq!(q.n_vars, 1);
    }

    #[test]
    fn hashjoin() {
        let mut db = Database::default();
        db.add_relation("r", 2)
            .insert(&[1, 2])
            .insert(&[3, 4])
            .insert(&[3, 5]);
        db.add_relation("s", 3)
            .insert(&[0, 1, 3])
            .insert(&[1, 1, 3])
            .insert(&[0, 0, 2]);

        let h = HashJoin {
            small_key: 0,
            big_key: 2,
            small: Scan::new("r", 2),
            big: Scan::new("s", 3),
        };

        assert_eq!(
            vec![
                vec![3, 4, 0, 1, 3],
                vec![3, 5, 0, 1, 3],
                vec![3, 4, 1, 1, 3],
                vec![3, 5, 1, 1, 3]
            ],
            h.eval_to_vecs(&db),
        );

        let nested1 = EqFilter {
            keys: (2, 3),
            inner: h.clone().into_dyn(),
        };

        assert_eq!(
            vec![vec![3, 4, 1, 1, 3], vec![3, 5, 1, 1, 3]],
            nested1.eval_to_vecs(&db),
        );

        let nested2 = HashJoin {
            small_key: 0,
            big_key: 2,
            small: Scan::new("r", 2),
            big: EqFilter {
                keys: (0, 1),
                inner: Scan::new("s", 3),
            },
        }
        .into_dyn();

        assert_eq!(nested1.eval_to_vecs(&db), nested2.eval_to_vecs(&db));
    }
}
