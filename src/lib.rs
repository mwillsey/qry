use std::{
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
    slice::ChunksExact,
};

use bumpalo::Bump;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

mod expr;
pub use expr::*;

mod util;
use util::*;

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq + Default + Display {}
impl<T> Data for T where T: Debug + Clone + Hash + Eq + Default + Display {}

#[derive(Debug, Clone)]
pub struct Database<S, T: Data> {
    pub relations: HashMap<(S, usize), Vec<T>>,
}

impl<S, T: Data> Default for Database<S, T> {
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

impl<V, S, T> Atom<V, S, T> {
    pub fn new(symbol: S, terms: Vec<Term<V, T>>) -> Self {
        Self {
            symbol,
            arity: terms.len(),
            terms,
        }
    }
}

impl<V: Eq + Clone, S, T> Atom<V, S, T> {
    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(v.clone()),
            _ => None,
        })
    }

    pub fn has_var(&self, v: &V) -> bool {
        self.vars().position(|x| &x == v).is_some()
    }
}

pub type VarMap<V> = Vec<V>;

#[derive(Debug, Clone)]
pub struct Query<V, S, T> {
    pub atoms: Vec<Atom<V, S, T>>,
    by_var: HashMap<V, Vec<usize>>,
}

impl<V: PartialEq, S: PartialEq, T: PartialEq> PartialEq for Query<V, S, T> {
    fn eq(&self, other: &Self) -> bool {
        self.atoms == other.atoms
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone,
    S: RelationSymbol,
    T: Data,
{
    pub fn new(atoms: Vec<Atom<V, S, T>>) -> Self {
        let mut new = Self {
            atoms,
            by_var: Default::default(),
        };

        for (i, atom) in new.atoms.iter().enumerate() {
            for v in atom.vars() {
                let is = new.by_var.entry(v).or_default();
                if is.last() != Some(&i) {
                    is.push(i)
                }
            }
        }

        new
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
    T: Data + 'static,
{
    // pub fn compile(&self) -> (VarMap<V>, Expr<S, T>) {
    //     let na = self.atoms.len();
    //     assert_ne!(na, 0);

    //     let mut exprs: Vec<(usize, usize, VarMap<V>, Expr<S, T>)> = self
    //         .atoms
    //         .iter()
    //         .map(|atom| {
    //             let mut used_vars: HashMap<V, Vec<usize>> = Default::default();
    //             let mut term_eqs: Vec<(T, usize)> = vec![];
    //             for (i, term) in atom.terms.iter().enumerate() {
    //                 match term {
    //                     Term::Variable(v) => used_vars.entry(v.clone()).or_default().push(i),
    //                     Term::Constant(c) => term_eqs.push((c.clone(), i)),
    //                 }
    //             }

    //             let mut var_map = VarMap::default();
    //             let mut var_eqs: Vec<(usize, usize)> = vec![];
    //             for (v, occurences) in used_vars {
    //                 var_map.insert(v, occurences[0]);
    //                 for pair in occurences.windows(2) {
    //                     var_eqs.push((pair[0], pair[1]))
    //                 }
    //             }

    //             let n_filters = var_eqs.len() + term_eqs.len();
    //             let expr = Expr::Scan {
    //                 relation: (atom.symbol.clone(), atom.arity),
    //                 var_eqs,
    //                 term_eqs,
    //             };
    //             (1, n_filters, var_map, expr)
    //         })
    //         .collect();

    //     while exprs.len() > 1 {
    //         let mut proposals = vec![];
    //         let mut iter = exprs.iter().enumerate();
    //         while let Some((i1, (d1, nf1, v1, _e1))) = iter.next() {
    //             for (i2, (d2, nf2, v2, _e2)) in iter.clone() {
    //                 let nf = nf1 + nf2;
    //                 let mut score = 0;
    //                 let mut var_map = VarMap::default();
    //                 let mut merge: Vec<Sided> = vec![];
    //                 for (v, i) in v1.iter() {
    //                     var_map.insert(v.clone(), merge.len());
    //                     merge.push(Sided::left(*i));
    //                 }
    //                 for (v, i) in v2.iter() {
    //                     if !v1.contains_key(v) {
    //                         var_map.insert(v.clone(), merge.len());
    //                         merge.push(Sided::right(*i));
    //                     } else {
    //                         score += 1;
    //                     }
    //                 }
    //                 proposals.push(((d1.max(d2) + 1, score, nf), var_map, merge, (i1, i2)));
    //             }
    //         }
    //         let ((depth, score, nf), var_map, merge, (i1, i2)) =
    //             proposals.iter().max_by_key(|p| p.0).unwrap().clone();
    //         // dbg!(score, nf);
    //         assert!(i1 < i2);
    //         // must remove i2 first to not mess up index
    //         let (_, nf1, v2, e2) = exprs.remove(i2);
    //         let (_, nf2, v1, e1) = exprs.remove(i1);
    //         let (key1, key2): (Vec<usize>, Vec<usize>) = v1
    //             .iter()
    //             .filter_map(|(v, i1)| v2.get(v).map(|i2| (i1, i2)))
    //             .unzip();
    //         assert_eq!(score, key1.len());
    //         let expr = Expr::Join {
    //             left: Keyed {
    //                 key: key1,
    //                 expr: Box::new(e1),
    //             },
    //             right: Keyed {
    //                 key: key2,
    //                 expr: Box::new(e2),
    //             },
    //             merge,
    //         };
    //         exprs.push((depth, nf, var_map, expr));
    //     }

    //     assert_eq!(exprs.len(), 1);
    //     let (_, _nf, varmap, expr) = exprs.pop().unwrap();
    //     (varmap, expr)
    // }
}

#[derive(Debug, Default)]
struct Trie<T: Display>(HashMap<T, Self>);

// impl<'bump, T: Display> Display for Trie<'bump, T> {
// }

// impl<'a, T: Display> Default for Trie<'a, T> {
//     fn default() -> Self {
//         Self(Default::default())
//     }
// }

// impl<'a, T: Data> Trie<'a, T> {
//     fn len(&self) -> usize {
//         self.0.len()
//     }

//     fn insert(&mut self, bump: &'a Bump, shuffle: &[usize], tuple: &[T]) {
//         debug_assert!(shuffle.len() <= tuple.len());
//         // debug_assert_eq!(shuffle.len(), tuple.len());
//         let mut trie = self;
//         for i in shuffle {
//             trie = trie.0.get_or_default(tuple[*i].clone(), bump);
//         }
//     }
// }

impl<T: Data> Trie<T> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&mut self, shuffle: &[usize], tuple: &[T]) {
        debug_assert!(shuffle.len() <= tuple.len());
        // debug_assert_eq!(shuffle.len(), tuple.len());
        let mut trie = self;
        for i in shuffle {
            trie = trie.0.entry(tuple[*i].clone()).or_default();
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalContext<S, T: Display> {
    cache: HashMap<(S, usize, Vec<Vec<usize>>), Rc<Trie<T>>>,
}

impl<S, T: Display> EvalContext<S, T> {
    pub fn clear(&mut self) {
        self.cache.clear()
    }
}

impl<S, T: Display> Default for EvalContext<S, T> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}

pub type Result = std::result::Result<(), ()>;

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone + Debug,
    S: RelationSymbol,
    T: Data,
{
    pub fn vars(&self, db: &Database<S, T>) -> VarMap<V> {
        let mut vars_occur: HashMap<V, usize> = HashMap::default();
        for var in self.atoms.iter().flat_map(|atom| atom.vars()) {
            let p = vars_occur.entry(var).or_default();
            *p += 1;
        }
        let mut vars_card: HashMap<V, usize> = HashMap::default();
        for atom in &self.atoms {
            let relation = db.get(&(atom.symbol.clone(), atom.arity));
            for (i, var) in atom.vars().enumerate() {
                let p = vars_card.entry(var).or_default();
                *p = std::cmp::min(*p, relation.len());
            }
        }

        let mut vars: Vec<V> = vars_occur.keys().cloned().collect();
        vars.sort_by(|s, t| {
            return (vars_card[s] < 100)
                .cmp(&(vars_card[t] < 100))
                .then_with(|| vars_occur[s].cmp(&vars_occur[t]).reverse())
                .then_with(|| vars_card[s].cmp(&vars_card[t]));
        });

        vars

        // let mut vars: HashMap<V, (usize, usize)> = HashMap::default();
        // for atom in &self.atoms {
        //     for var in atom.vars() {
        //         let p = vars.entry(var).or_default();
        //         p.0 += 1;
        //         p.1 += 1;
        //     }
        // }

        // let mut varmap = HashMap::default();
        // loop {
        //     let biggest = vars.iter().max_by_key(|(_v, counts)| *counts);
        //     let v = match biggest {
        //         Some((v, _counts)) => v.clone(),
        //         None => break,
        //     };

        //     vars.remove(&v);

        //     let i = varmap.len();
        //     varmap.insert(v, i);

        //     vars.values_mut().for_each(|counts| counts.0 = 0);
        //     for (v, counts) in vars.iter_mut() {
        //         for atom in &self.atoms {
        //             if atom.has_var(v) {
        //                 counts.0 += 1;
        //             }
        //         }
        //     }
        // }
        // varmap
    }

    pub fn join<F>(
        &self,
        varmap: &VarMap<V>,
        db: &Database<S, T>,
        ctx: &mut EvalContext<S, T>,
        mut f: F,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
    {
        let vars: Vec<_> = varmap
            .iter()
            .map(|v| (v.clone(), self.by_var[v].clone()))
            .collect();

        let mut tries: Vec<Rc<Trie<T>>> = Vec::with_capacity(self.atoms.len());
        for atom in &self.atoms {
            let mut shuffle = Vec::with_capacity(atom.terms.len());
            let mut index = Vec::with_capacity(atom.terms.len());
            for (var, _) in &vars {
                let mut found_var = false;
                for (i, term) in atom.terms.iter().enumerate() {
                    if let Term::Variable(v) = term {
                        if var == v && !index.contains(&i) {
                            if found_var == false {
                                found_var = true;
                                index.push(i);
                                shuffle.push(vec![i]);
                            } else {
                                shuffle.last_mut().unwrap().push(i);
                            }
                        }
                    }
                }
            }
            assert!(shuffle.len() <= atom.terms.len());

            let key = (atom.symbol.clone(), atom.arity, shuffle.clone());
            // only keep meaningful constraints 
            shuffle.retain(|group| group.len() > 1);
            let trie = ctx
                .cache
                .entry(key)
                .or_insert_with(|| {
                    let mut trie = Trie::default();
                    for tuple in db.get(&(atom.symbol.clone(), atom.arity)) {
                        if shuffle.iter().all(|group| group[1..].iter().all(|i| tuple[*i] == tuple[group[0]])) {
                            trie.insert(&index, tuple);
                        }
                    }
                    Rc::new(trie)
                })
                .clone();

            tries.push(trie)
        }

        let mut tries: Vec<&Trie<T>> = tries.iter().map(|t| t.as_ref()).collect();

        let empty = Trie::default();
        self.gj(&mut f, &mut vec![], &vars, &mut tries, &empty)
    }

    #[inline]
    fn gj_impl_base<'a, F>(
        &self,
        f: &mut F,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&Trie<T>],
        _empty: &Trie<T>,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
    {
        let pos = tuple.len();
        assert!(pos < vars.len());

        let (x, js) = &vars[pos];
        debug_assert!(js.iter().all(|&j| self.atoms[j].has_var(x)));

        match js.len() {
            1 => {
                let j = js[0];
                tuple.push(Default::default());
                for val in relations[j].0.keys() {
                    tuple[pos] = val.clone();
                    f(tuple)?;
                }
                tuple.pop();
            }
            2 => {
                let (j_min, j_max) = if relations[js[0]].len() < relations[js[1]].len() {
                    (js[0], js[1])
                } else {
                    (js[1], js[0])
                };
                let r = relations[j_min];
                let rj = relations[j_max];
                let intersection = r.0.keys().filter(|t| rj.0.contains_key(t));
                tuple.push(Default::default());
                for val in intersection {
                    tuple[pos] = val.clone();
                    f(tuple)?;
                }
                tuple.pop();
            }
            _ => {
                let j_min = js
                    .iter()
                    .copied()
                    .min_by_key(|j| relations[*j].len())
                    .unwrap();

                let mut intersection: Vec<T> = relations[j_min].0.keys().cloned().collect();
                for &j in js {
                    if j != j_min {
                        let rj = &relations[j].0;
                        intersection.retain(|t| rj.contains_key(t));
                    }
                }

                let pos = tuple.len();
                tuple.push(Default::default());
                for val in intersection {
                    tuple[pos] = val;
                    f(tuple)?;
                }
                tuple.pop();
            }
        };
        Ok(())
    }

    #[inline]
    fn gj_impl<'a, This>(
        &self,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&'a Trie<T>],
        empty: &'a Trie<T>,
        mut this: This,
    ) -> Result
    where
        This: FnMut(&mut Vec<T>, &mut [&'a Trie<T>]) -> Result,
    {
        let pos = tuple.len();
        assert!(pos < vars.len());

        let (x, js) = &vars[pos];
        debug_assert!(js.iter().all(|&j| self.atoms[j].has_var(x)));

        match js.len() {
            1 => {
                let j = js[0];
                let r = relations[j];
                tuple.push(Default::default());
                for val in relations[j].0.keys() {
                    relations[j] = r.0.get(&val).unwrap_or(empty);
                    tuple[pos] = val.clone();
                    this(tuple, relations)?;
                }
                tuple.pop();
                relations[j] = r;
            }
            2 => {
                let (j_min, j_max) = if relations[js[0]].len() < relations[js[1]].len() {
                    (js[0], js[1])
                } else {
                    (js[1], js[0])
                };
                let r = relations[j_min];
                let rj = relations[j_max];
                let intersection = r.0.keys().filter(|t| rj.0.contains_key(t));
                tuple.push(Default::default());
                for val in intersection {
                    relations[j_min] = r.0.get(&val).unwrap_or(empty);
                    relations[j_max] = rj.0.get(&val).unwrap_or(empty);
                    tuple[pos] = val.clone();
                    this(tuple, relations)?;
                }
                tuple.pop();
                relations[j_min] = r;
                relations[j_max] = rj;
            }
            _ => {
                let j_min = js
                    .iter()
                    .copied()
                    .min_by_key(|j| relations[*j].len())
                    .unwrap();

                let mut intersection: Vec<T> = relations[j_min].0.keys().cloned().collect();
                for &j in js {
                    if j != j_min {
                        let rj = &relations[j].0;
                        intersection.retain(|t| rj.contains_key(t));
                    }
                }
                let jrelations: Vec<_> = js.iter().map(|&j| relations[j]).collect();
                let pos = tuple.len();
                tuple.push(Default::default());
                for val in intersection {
                    for (&j, r) in js.iter().zip(&jrelations) {
                        let sub_r = r.0.get(&val).unwrap_or(empty);
                        relations[j] = sub_r;
                    }
                    tuple[pos] = val;
                    this(tuple, relations)?;
                }
                tuple.pop();
                for (&j, r) in js.iter().zip(&jrelations) {
                    relations[j] = r
                }
            }
        };
        Ok(())
    }

    fn gj<'a, F>(
        &self,
        f: &mut F,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&'a Trie<T>],
        empty: &'a Trie<T>,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
    {
        let rem = vars.len() - tuple.len() - 1;
        match rem {
            0 => {
                self.gj_impl_base(f, tuple, vars, relations, empty)
            }
            1 => {
                self.gj_impl(tuple, vars, relations, empty, |tuple, relations| {
                    self.gj_impl_base(f, tuple, vars, relations, empty)
                })
            }
            _ => self.gj_impl(tuple, vars, relations, empty, |tuple, relations| {
                self.gj(f, tuple, vars, relations, empty)
            }),
        }
    }
}
