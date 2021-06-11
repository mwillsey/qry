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
        self.vars().any(|x| &x == v)
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

#[derive(Debug, Default)]
struct Index<T> {
    buf: Vec<T>,
    trie: HashMap<T, Self>,
}

impl<T: Data> Index<T> {
    fn map_len(&self) -> usize {
        self.trie.len()
    }

    fn intersect_keys<'a, const N: usize>(
        &'a self,
        others: [&'a Self; N],
    ) -> impl Iterator<Item = T> + 'a {
        self.trie
            .keys()
            .cloned()
            .filter(move |k| others.iter().all(|idx| idx.trie.contains_key(k)))
    }

    fn insert(&mut self, trie_path: &[usize], buf_path: &[usize], tuple: &[T]) {
        let mut index = self;
        for i in trie_path {
            index = index.trie.entry(tuple[*i].clone()).or_default();
        }
        for i in buf_path {
            index.buf.push(tuple[*i].clone());
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalContext<S, T: Display> {
    cache: HashMap<(S, usize, Vec<Vec<usize>>, usize), Arc<Index<T>>>,
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
            .filter_map(|(v, occur)| (*occur > 1).then(|| v))
            .cloned()
            .collect();
        vars.sort_by(|s, t| {
            (vars_card[s] < 100)
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
        f: F,
    ) -> Result
    where
        F: FnMut(&[T]) -> Result,
        T: Data,
    {
        let vars: Vec<_> = varmap
            .iter()
            .map(|v| (v.clone(), self.by_var[v].as_slice()))
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
                        if !found_var {
                            found_var = true;
                            access_path.push(i);
                            shuffle.push(vec![i]);
                        } else {
                            shuffle.last_mut().unwrap().push(i);
                        }
                    }
                }
            }
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

            let key = (atom.symbol.clone(), atom.arity, shuffle.clone(), break_at);
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
                                trie.insert(trie_path, buf_path, tuple);
                            }
                        }
                        Arc::new(trie)
                    })
                    .clone();

            break_ats.push(break_at);
            chunk_sizes.push(buf_path.len());
            tries.push(trie);
        }

        // TODO right now this is banking on some invariants from the
        // variable ordering (all non-intersections are last and grouped)

        let mut intersection_groups = vec![];
        for (_v, occurs) in &vars {
            if occurs.len() > 1 {
                intersection_groups.push(*occurs);
            } else {
                break;
            }
        }

        let tail_vars: HashSet<V> = vars[intersection_groups.len()..]
            .iter()
            .map(|(v, _)| v.clone())
            .collect();
        let batches = self
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                let has_tail_var = a.vars().any(|v| tail_vars.contains(&v));
                has_tail_var.then(|| Batch {
                    relation: i,
                    chunk_size: chunk_sizes[i],
                })
            })
            .collect();

        VM {
            f,
            intersection_groups,
            batches,
            tuple: vec![T::default(); self.by_var.len()],
            relations: tries.iter().map(|t| t.as_ref()).collect(),
        }
        .step(0)
    }
}

#[derive(Copy, Clone)]
struct Batch {
    relation: usize,
    chunk_size: usize,
}

struct VM<'a, F, T> {
    f: F,
    tuple: Vec<T>,
    intersection_groups: Vec<&'a [usize]>,
    batches: Vec<Batch>,
    relations: Vec<&'a Index<T>>,
}

impl<'a, F, T> VM<'a, F, T>
where
    T: Data,
    F: FnMut(&[T]) -> Result,
{
    fn step(&mut self, depth: usize) -> Result {
        macro_rules! inner {
            ($intersection:expr, $js:ident, $rs:ident) => {{
                let intersection = $intersection;
                if depth + 1 == self.tuple.len() {
                    for val in intersection {
                        self.tuple[depth] = val;
                        (self.f)(&self.tuple)?;
                    }
                } else {
                    for val in intersection {
                        for (&j, r) in $js.iter().zip($rs.iter()) {
                            self.relations[j] = r.trie.get(&val).unwrap();
                        }
                        self.tuple[depth] = val;
                        self.step(depth + 1)?
                    }
                    for (&j, r) in $js.iter().zip($rs.iter()) {
                        self.relations[j] = r;
                    }
                }

                Ok(())
            }};
        }

        if depth >= self.intersection_groups.len() {
            return self.tail(depth, 0);
        }

        let js = self.intersection_groups[depth];

        match js.len() {
            0 => unreachable!(),
            1 => {
                let rs = [self.relations[js[0]]];
                let intersection = rs[0].trie.keys().cloned();
                inner!(intersection, js, rs)
            }
            2 => {
                let rs = [self.relations[js[0]], self.relations[js[1]]];

                let intersection = if rs[0].map_len() <= rs[1].map_len() {
                    rs[0].intersect_keys([rs[1]])
                } else {
                    rs[1].intersect_keys([rs[0]])
                };
                inner!(intersection, js, rs)
            }
            3 => {
                let rs = [
                    self.relations[js[0]],
                    self.relations[js[1]],
                    self.relations[js[2]],
                ];
                let is_smaller = |i: usize, j: usize, k: usize| {
                    rs[i].map_len() <= rs[j].map_len() && rs[i].map_len() <= rs[k].map_len()
                };

                let intersection = if is_smaller(0, 1, 2) {
                    rs[0].intersect_keys([rs[1], rs[2]])
                } else if is_smaller(1, 0, 2) {
                    rs[1].intersect_keys([rs[0], rs[2]])
                } else {
                    rs[2].intersect_keys([rs[0], rs[1]])
                };
                inner!(intersection, js, rs)
            }
            _ => {
                let j_min = js
                    .iter()
                    .copied()
                    .min_by_key(|j| self.relations[*j].map_len())
                    .unwrap();

                let mut intersection: Vec<T> = self.relations[j_min].trie.keys().cloned().collect();

                for &j in js {
                    if j != j_min {
                        let rj = &self.relations[j].trie;
                        intersection.retain(|t| rj.contains_key(t));
                    }
                }

                let rs: Vec<_> = js.iter().map(|&j| self.relations[j]).collect();
                inner!(intersection, js, rs)
            }
        }
    }

    fn tail(&mut self, depth: usize, batch_i: usize) -> Result {
        assert!(batch_i < self.batches.len());
        let batch = self.batches[batch_i];
        let relation = self.relations[batch.relation];

        if batch_i + 1 == self.batches.len() {
            for chunk in relation.buf.chunks_exact(batch.chunk_size) {
                self.tuple[depth..].clone_from_slice(chunk);
                (self.f)(&self.tuple)?;
            }
        } else {
            for chunk in relation.buf.chunks_exact(batch.chunk_size) {
                let next_depth = depth + batch.chunk_size;
                self.tuple[depth..next_depth].clone_from_slice(chunk);
                self.tail(next_depth, batch_i + 1)?;
            }
        }

        Ok(())
    }
}
