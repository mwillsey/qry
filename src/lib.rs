use fxhash::FxHashMap;
use std::{fmt::Debug, hash::Hash, slice::ChunksExact};

mod expr;
pub use expr::*;

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq {}
impl<T> Data for T where T: Debug + Clone + Hash + Eq {}

#[derive(Debug, Clone)]
pub struct Database<S, T> {
    pub relations: FxHashMap<(S, usize), Vec<T>>,
}

impl<S, T> Default for Database<S, T> {
    fn default() -> Self {
        let relations = Default::default();
        Self { relations }
    }
}

impl<S: RelationSymbol, T: Data> Database<S, T> {
    pub fn add_relation_with_data(&mut self, sym: S, arity: usize, data: Vec<T>) {
        self.relations.insert((sym, arity), data);
    }

    pub fn get(&self, key: &(S, usize)) -> ChunksExact<T> {
        if let Some(data) = self.relations.get(key) {
            data.chunks_exact(key.1)
        } else {
            (&[]).chunks_exact(1)
        }
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
    pub arity: usize,
    pub terms: Vec<Term<V, T>>,
}

impl<V: Clone, S, T> Atom<V, S, T> {
    pub fn new(symbol: S, terms: Vec<Term<V, T>>) -> Self {
        Self {
            symbol,
            arity: terms.len(),
            terms,
        }
    }

    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(v.clone()),
            _ => None,
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
    pub fn compile(self) -> Expr<S, T> {
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

                Expr::Scan {
                    relation: (atom.symbol, atom.arity),
                    var_eqs,
                    term_eqs,
                }
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

                let merge: Vec<Sided> = my_vars
                    .into_iter()
                    .map(|v| {
                        if let Some(i) = q1.index_of(&v) {
                            Sided::left(i)
                        } else if let Some(i) = q2.index_of(&v) {
                            Sided::right(i)
                        } else {
                            unreachable!("var has to come from one side")
                        }
                    })
                    .collect();

                Expr::Join {
                    left: Keyed {
                        key: key1,
                        expr: Box::new(q1.compile()),
                    },
                    right: Keyed {
                        key: key2,
                        expr: Box::new(q2.compile()),
                    },
                    merge,
                }
            }
        }
    }
}
