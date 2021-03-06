use fxhash::{FxHashMap, FxHashSet};
use std::{fmt::Debug, hash::Hash, rc::Rc, slice::ChunksExact};

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

pub type VarMap<V> = FxHashMap<V, usize>;

#[derive(Debug, Clone, PartialEq)]
pub struct Query<V, S, T> {
    pub atoms: Vec<Atom<V, S, T>>,
}

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone,
    S: RelationSymbol,
    T: Data,
{
    pub fn new(atoms: Vec<Atom<V, S, T>>) -> Self {
        Self { atoms }
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
    T: Data + 'static,
{
    pub fn compile(&self) -> (VarMap<V>, Expr<S, T>) {
        let na = self.atoms.len();
        assert_ne!(na, 0);

        let mut exprs: Vec<(usize, usize, VarMap<V>, Expr<S, T>)> = self.atoms.iter()
            .map(|atom| {
                let mut used_vars: FxHashMap<V, Vec<usize>> = Default::default();
                let mut term_eqs: Vec<(T, usize)> = vec![];
                for (i, term) in atom.terms.iter().enumerate() {
                    match term {
                        Term::Variable(v) => used_vars.entry(v.clone()).or_default().push(i),
                        Term::Constant(c) => term_eqs.push((c.clone(), i)),
                    }
                }

                let mut var_map = VarMap::default();
                let mut var_eqs: Vec<(usize, usize)> = vec![];
                for (v, occurences) in used_vars {
                    var_map.insert(v, occurences[0]);
                    for pair in occurences.windows(2) {
                        var_eqs.push((pair[0], pair[1]))
                    }
                }

                let n_filters = var_eqs.len() + term_eqs.len();
                let expr = Expr::Scan {
                    relation: (atom.symbol.clone(), atom.arity),
                    var_eqs,
                    term_eqs,
                };
                (1, n_filters, var_map, expr)
            })
            .collect();

        while exprs.len() > 1 {
            let mut proposals = vec![];
            let mut iter = exprs.iter().enumerate();
            while let Some((i1, (d1, nf1, v1, _e1))) = iter.next() {
                for (i2, (d2, nf2, v2, _e2)) in iter.clone() {
                    let nf = nf1 + nf2;
                    let mut score = 0;
                    let mut var_map = VarMap::default();
                    let mut merge: Vec<Sided> = vec![];
                    for (v, i) in v1.iter() {
                        var_map.insert(v.clone(), merge.len());
                        merge.push(Sided::left(*i));
                    }
                    for (v, i) in v2.iter() {
                        if !v1.contains_key(v) {
                            var_map.insert(v.clone(), merge.len());
                            merge.push(Sided::right(*i));
                        } else {
                            score += 1;
                        }
                    }
                    proposals.push(((d1.max(d2) + 1, score, nf), var_map, merge, (i1, i2)));
                }
            }
            let ((depth, score, nf), var_map, merge, (i1, i2)) = proposals.iter().max_by_key(|p| p.0).unwrap().clone();
            dbg!(score, nf);
            assert!(i1 < i2);
            // must remove i2 first to not mess up index
            let (_, nf1, v2, e2) = exprs.remove(i2);
            let (_, nf2, v1, e1) = exprs.remove(i1);
            let (key1, key2): (Vec<usize>, Vec<usize>) = v1
                .iter()
                .filter_map(|(v, i1)| v2.get(v).map(|i2| (i1, i2)))
                .unzip();
            assert_eq!(score, key1.len());
            let expr = Expr::Join {
                left: Keyed {
                    key: key1,
                    expr: Box::new(e1),
                },
                right: Keyed {
                    key: key2,
                    expr: Box::new(e2),
                },
                merge,
            };
            exprs.push((depth, nf, var_map, expr));
        }

        println!("Compiled");

        assert_eq!(exprs.len(), 1);
        let (_, _nf, varmap, expr) = exprs.pop().unwrap();
        (varmap, expr)
    }
}
