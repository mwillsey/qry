#![allow(unused_imports)]
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    slice::ChunksExact,
    sync::Arc,
};

use bumpalo::Bump;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

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
pub struct Term<V>(pub V);

#[derive(Debug, Clone, PartialEq)]
pub struct Atom<V, S> {
    pub symbol: S,
    pub arity: usize,
    pub terms: Vec<Term<V>>,
}

impl<V, S> Atom<V, S> {
    pub fn new(symbol: S, terms: Vec<Term<V>>) -> Self {
        Self {
            symbol,
            arity: terms.len(),
            terms,
        }
    }
}

impl<V: Eq + Clone, S> Atom<V, S> {
    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().map(|t| t.0.clone())
    }

    pub fn has_var(&self, v: &V) -> bool {
        self.vars().position(|x| &x == v).is_some()
    }
}

pub type VarMap<V> = Vec<V>;

#[derive(Debug, Clone)]
pub struct Query<V, S> {
    pub atoms: Vec<Atom<V, S>>,
    by_var: HashMap<V, Vec<usize>>,
}

impl<V: PartialEq, S: PartialEq> PartialEq for Query<V, S> {
    fn eq(&self, other: &Self) -> bool {
        self.atoms == other.atoms
    }
}

impl<V, S> Query<V, S>
where
    V: Eq + Hash + Clone,
    S: RelationSymbol,
{
    pub fn new(atoms: Vec<Atom<V, S>>) -> Self {
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

impl<V, S> Query<V, S>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
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
struct Index<T: Display> {
    buf: Vec<T>,
    trie: HashMap<T, Self>,
}

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

struct AccessPath<'a> {
    trie_path: &'a [usize],
    buf_path: &'a [usize],
}

impl<T: Data> Index<T> {
    fn map_len(&self) -> usize {
        self.trie.len()
    }

    fn insert(&mut self, shuffle: &AccessPath<'_>, tuple: &[T])
    {
        let mut index = self;
        for i in shuffle.trie_path {
            index = index.trie.entry(tuple[*i].clone()).or_default();
        }
        for i in shuffle.buf_path {
            index.buf.push(tuple[*i].clone());
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalContext<S, T: Display> {
    cache: HashMap<(S, usize, Vec<Vec<usize>>), Arc<Index<T>>>,
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

type Result = std::result::Result<(), ()>;

impl<V, S> Query<V, S>
where
    V: Eq + Hash + Clone + Debug,
    S: RelationSymbol,
{
    pub fn vars<T: Data>(&self, db: &Database<S, T>) -> VarMap<V> {
        let mut vars_occur: HashMap<V, usize> = HashMap::default();
        for var in self.atoms.iter().flat_map(|atom| atom.vars()) {
            let p = vars_occur.entry(var).or_default();
            *p += 1;
        }
        let mut vars_card: HashMap<V, usize> = HashMap::default();
        for atom in &self.atoms {
            let relation = db.get(&(atom.symbol.clone(), atom.arity));
            for var in atom.vars() {
                let p = vars_card.entry(var).or_default();
                *p = std::cmp::min(*p, relation.len());
            }
        }

        // first only consider variables with multi occurrence
        let mut vars: Vec<V> = vars_occur
            .iter()
            .filter_map(|(v, occur)| if *occur > 1 { Some(v) } else { None })
            .cloned()
            .collect();
        vars.sort_by(|s, t| {
            return (vars_card[s] < 100)
                .cmp(&(vars_card[t] < 100))
                .then_with(|| vars_occur[s].cmp(&vars_occur[t]).reverse())
                .then_with(|| vars_card[s].cmp(&vars_card[t]))
        });

        // next add variables with only one occurrence, so they are
        // batched together by the relation they are in
        for atom in &self.atoms {
            for var in atom.vars() {
                if vars_occur[&var] == 1 {
                    vars.push(var);
                }
            }
        }

        vars
    }

    pub fn join<F, T>(
        &self,
        varmap: &VarMap<V>,
        db: &Database<S, T>,
        ctx: &mut EvalContext<S, T>,
        mut f: F,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
        T: Data,
    {
        let vars: Vec<_> = varmap
            .iter()
            .map(|v| (v.clone(), self.by_var[v].clone()))
            .collect();

        let mut tries: Vec<Arc<Index<T>>> = Vec::with_capacity(self.atoms.len());
        let mut break_ats: Vec<usize> = Vec::with_capacity(self.atoms.len());
        let mut chunk_sizes: Vec<usize> = Vec::with_capacity(self.atoms.len());
        for atom in &self.atoms {
            let mut shuffle = Vec::with_capacity(atom.terms.len());
            let mut access_path = Vec::with_capacity(atom.terms.len());
            for (var, _) in &vars {
                let mut found_var = false;
                for (i, term) in atom.terms.iter().enumerate() {
                    if var == &term.0 && !access_path.contains(&i) {
                        if found_var == false {
                            found_var = true;
                            access_path.push(i);
                            shuffle.push(vec![i]);
                        } else {
                            shuffle.last_mut().unwrap().push(i);
                        }
                    }
                }
            }
            let chunk_size = access_path.len();
            let break_at = access_path
                .iter()
                .rev()
                .position(|i| {
                    let var = &atom.vars().collect::<Vec<_>>()[*i];
                    let (_var, by_var) = vars
                        .iter()
                        .find(|(var1, _)| var1 == var)
                        .expect("variable not found");
                    // TODO: Note patterns like R(x, x) are not batched
                    by_var.len() != 1
                })
                .map(|rlen| access_path.len() - rlen)
                .unwrap_or(0);
            let (trie_path, buf_path) = access_path.split_at(break_at);
            let access_path = AccessPath {
                trie_path: trie_path,
                buf_path: buf_path,
            };

            let key = (atom.symbol.clone(), atom.arity, shuffle.clone());
            // only keep meaningful constraints
            shuffle.retain(|group| group.len() > 1);
            let trie =
                ctx.cache
                    .entry(key)
                    .or_insert_with(|| {
                        let mut trie = Index::default();
                        for tuple in db.get(&(atom.symbol.clone(), atom.arity)) {
                            if shuffle.iter().all(|group| {
                                group[1..].iter().all(|i| tuple[*i] == tuple[group[0]])
                            }) {
                                trie.insert(&access_path, tuple);
                            }
                        }
                        Arc::new(trie)
                    })
                    .clone();

            break_ats.push(break_at);
            chunk_sizes.push(chunk_size);
            tries.push(trie);
        }

        let mut tries: Vec<&Index<T>> = tries
            .iter()
            .map(|t| t.as_ref())
            .collect();

        let empty = Index::default();
        self.gj(&mut f, &mut vec![], &vars, &mut tries, &mut break_ats, &chunk_sizes, &empty)
    }

    #[inline]
    fn gj_impl_base<'a, F, T>(
        &self,
        f: &mut F,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&Index<T>],
        _empty: &Index<T>,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
        T: Data,
    {
        let pos = tuple.len();
        assert!(pos < vars.len());

        let (x, js) = &vars[pos];
        debug_assert!(js.iter().all(|&j| self.atoms[j].has_var(x)));
        match js.len() {
            1 => {
                // no need to implement batching for the base case
                // because there's always only one variable 
                // remaining in the base case.
                let j = js[0];
                tuple.push(Default::default());
                for val in relations[j].trie.keys() {
                    tuple[pos] = val.clone();
                    f(tuple)?;
                }
                tuple.pop();
            }
            2 => {
                let (j_min, j_max) = if relations[js[0]].map_len() < relations[js[1]].map_len() {
                    (js[0], js[1])
                } else {
                    (js[1], js[0])
                };
                let r = relations[j_min];
                let rj = relations[j_max];
                let intersection = r.trie.keys().filter(|t| rj.trie.contains_key(t));
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
                    .min_by_key(|j| relations[*j].map_len())
                    .unwrap();

                let mut intersection: Vec<T> = relations[j_min].trie.keys().cloned().collect();
                for &j in js {
                    if j != j_min {
                        let rj = &relations[j].trie;
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
    fn gj_impl<'a, T, This>(
        &self,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&'a Index<T>],
        break_ats: &mut [usize],
        chunk_sizes: &[usize],
        empty: &'a Index<T>,
        mut this: This,
    ) -> Result
    where
        This: FnMut(&mut Vec<T>, &mut [&'a Index<T>], &mut [usize]) -> Result,
        T: Data,
    {
        let pos = tuple.len();
        if !(pos < vars.len()) {
            eprintln!("ERROR: pos: {}, vars.len(): {}", pos, vars.len());
            eprintln!("{:?} {:?}", tuple, vars);
        }
        assert!(pos < vars.len());

        let (x, js) = &vars[pos];
        debug_assert!(js.iter().all(|&j| self.atoms[j].has_var(x)));

        match js.len() {
            1 => {
                let j = js[0];
                if break_ats[j] == 0 {
                    // TODO: pattern match and unroll on the size
                    let len = tuple.len();
                    tuple.resize(len + chunk_sizes[j], Default::default());
                    for chunk in relations[j].buf.chunks_exact(chunk_sizes[j]) {
                        for i in 0..chunk_sizes[j] {
                            tuple[len + i] = chunk[i].clone();
                        }
                        this(tuple, relations, break_ats)?;
                    }
                    tuple.truncate(len);
                } else {
                    let r = relations[j];
                    tuple.push(Default::default());
                    break_ats[j] -= 1;
                    for val in relations[j].trie.keys() {
                        relations[j] = r.trie.get(&val).unwrap_or(empty);
                        tuple[pos] = val.clone();
                        this(tuple, relations, break_ats)?;
                    }
                    break_ats[j] += 1;
                    tuple.pop();
                    relations[j] = r;
                }
            }
            2 => {
                let (j_min, j_max) = if relations[js[0]].map_len() < relations[js[1]].map_len() {
                    (js[0], js[1])
                } else {
                    (js[1], js[0])
                };
                let r = relations[j_min];
                let rj = relations[j_max];
                let intersection = r.trie.keys().filter(|t| rj.trie.contains_key(t));
                tuple.push(Default::default());
                break_ats[j_min] -= 1;
                break_ats[j_max] -= 1;
                for val in intersection {
                    relations[j_min] = r.trie.get(&val).unwrap_or(empty);
                    relations[j_max] = rj.trie.get(&val).unwrap_or(empty);
                    tuple[pos] = val.clone();
                    this(tuple, relations, break_ats)?;
                }
                tuple.pop();
                break_ats[j_min] += 1;
                break_ats[j_max] += 1;
                relations[j_min] = r;
                relations[j_max] = rj;
            }
            _ => {
                let j_min = js
                    .iter()
                    .copied()
                    .min_by_key(|j| relations[*j].map_len())
                    .unwrap();

                let mut intersection: Vec<T> = relations[j_min].trie.keys().cloned().collect();
                for &j in js {
                    if j != j_min {
                        let rj = &relations[j].trie;
                        intersection.retain(|t| rj.contains_key(t));
                    }
                }
                let jrelations: Vec<_> = js.iter().map(|&j| relations[j]).collect();
                let pos = tuple.len();
                tuple.push(Default::default());
                js.iter().for_each(|j| break_ats[*j] -= 1);
                for val in intersection {
                    for (&j, r) in js.iter().zip(&jrelations) {
                        let sub_r = r.trie.get(&val).unwrap_or(empty);
                        relations[j] = sub_r;
                    }
                    tuple[pos] = val;
                    this(tuple, relations, break_ats)?;
                }
                tuple.pop();
                for (&j, r) in js.iter().zip(&jrelations) {
                    break_ats[j] += 1;
                    relations[j] = r;
                }
            }
        };
        Ok(())
    }

    fn gj<'a, T, F>(
        &self,
        f: &mut F,
        tuple: &mut Vec<T>,
        vars: &[(V, Vec<usize>)],
        relations: &mut [&'a Index<T>],
        break_ats: &mut [usize],
        chunk_sizes: &[usize],
        empty: &'a Index<T>,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
        T: Data,
    {
        let rem = vars.len() - tuple.len() - 1;
        match rem {
            0 => self.gj_impl_base(f, tuple, vars, relations, empty),
            1 => self.gj_impl(tuple, vars, relations, break_ats, chunk_sizes, empty, |tuple, relations, _break_ats| {
                self.gj_impl_base(f, tuple, vars, relations, empty)
            }),
            _ => self.gj_impl(tuple, vars, relations, break_ats, chunk_sizes, empty, |tuple, relations, break_ats| {
                self.gj(f, tuple, vars, relations, break_ats, chunk_sizes, empty)
            }),
        }
    }
}
