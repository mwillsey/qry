#![allow(dead_code)]
use std::hash::{BuildHasherDefault, Hash, Hasher};
use bumpalo::Bump;
use std::fmt::Display;

macro_rules! unreachable_unchecked {
    () => {
        if cfg!(debug_assertions) {
            unreachable!()
        } else {
            unsafe { std::hint::unreachable_unchecked() }
        }
    };
}

macro_rules! probe_mut {
    ($self:ident, $k:expr, {
        Ok($v:pat) => $v_expr:expr,
        Err($spot:pat) => $spot_expr:expr $(,)?
    }) => {{
        let cap = $self.capacity();
        let mut i = $self.hash($k) as usize % cap;
        let ptr = $self.spots.as_mut_ptr();
        loop {
            let r = unsafe { &mut *(ptr.offset(i as isize)) };
            #[allow(unreachable_code)]
            match r {
                Some((k2, $v)) if $k == k2 => break $v_expr,
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                _ => {
                    let $spot = i;
                    break $spot_expr;
                }
            }
        }
    }};
}

#[derive(Debug)]
pub struct BumpHashMap<
    'bump,
    K: Display + 'bump,
    V: Display + 'bump,
    S = BuildHasherDefault<rustc_hash::FxHasher>,
> {
    len: usize,
    spots: &'bump mut [Option<(K, V)>],
    builder: S,
}

impl<'bump, K: 'bump + Display, V: 'bump + Display> Default for BumpHashMap<'bump, K, V> {
    fn default() -> Self {
        Self {
            len: 0,
            spots: &mut [],
            builder: Default::default(),
        }
    }
}

impl<'bump, K, V, S> BumpHashMap<'bump, K, V, S>
where
    K: 'bump + Hash + Eq + Display,
    V: 'bump + Display,
    S: std::hash::BuildHasher,
{
    const GROWTH_FACTOR: usize = 2;
    const MIN_CAPACITY: usize = 4;

    pub fn hash(&self, k: &K) -> u64 {
        let mut hasher = self.builder.build_hasher();
        k.hash(&mut hasher);
        hasher.finish()
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.get(k).is_some()
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.spots
            .iter()
            .filter_map(|x| x.as_ref())
            .map(|tup| &tup.0)
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let cap = self.capacity();
        if cap == 0 {
            return None;
        }
        let mut i = self.hash(k) as usize % cap;
        loop {
            match unsafe { self.spots.get_unchecked(i) } {
                Some((k2, v)) if k == k2 => return Some(v),
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                _ => return None,
            }
        }
    }

    pub fn get_or_default(&mut self, k: K, bump: &'bump Bump) -> &mut V
    where
        V: Default,
    {
        if self.capacity() == 0 {
            self.grow(bump);
        }

        probe_mut!(self, &k, {
            Ok(v) => v,
            Err(mut spot) => {
                self.len += 1;
                if self.is_loaded() {
                    self.grow(bump);
                    spot = probe_mut!(self, &k, {
                        Ok(_) => unreachable_unchecked!(),
                        Err(spot) => spot,
                    });
               }
                self.set(spot, k, V::default())
            }
        })
    }

    // TODO: comment out for now since it's preventing clippy from passing
    // fn probe_mut(&mut self, k: &K) -> Result<&mut V, &mut Option<(K, V)>> {
    //     match self.probe(k) {
    //         Ok(v) => Ok(unsafe { &mut *(v as *const V as *mut V) }),
    //         Err(spot) => Ok(unsafe { &mut *(spot as *const _ as *mut _) }),
    //     }
    // }

    fn probe(&self, k: &K) -> Result<&V, &Option<(K, V)>> {
        let cap = self.capacity();
        let mut i = self.hash(k) as usize % cap;
        loop {
            debug_assert!(i < self.spots.len());
            match unsafe { self.spots.get_unchecked(i) } {
                Some((k2, v)) if k == k2 => return Ok(v),
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                spot => return Err(spot),
            }
        }
    }

    fn grow(&mut self, bump: &'bump Bump) {
        debug_assert!(self.is_loaded());
        let new_cap = match self.capacity() {
            0 => Self::MIN_CAPACITY,
            cap => cap * Self::GROWTH_FACTOR,
        };
        let new_spots = bump.alloc_slice_fill_default(new_cap);
        let old_spots = std::mem::replace(&mut self.spots, new_spots);
        for spot in old_spots.iter_mut() {
            if let Some((k, v)) = spot.take() {
                probe_mut!(self, &k, {
                    Ok(_) => unreachable_unchecked!(),
                    Err(new_spot) => self.set(new_spot, k, v),
                });
            }
        }
    }

    fn set(&mut self, i: usize, k: K, v: V) -> &mut V {
        let spot = unsafe { self.spots.get_unchecked_mut(i) };
        *spot = Some((k, v));
        &mut spot.as_mut().unwrap().1
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.spots.len()
    }

    fn is_loaded(&self) -> bool {
        self.len() * 4 >= self.capacity() * 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap() {
        let bump = &Bump::new();
        let mut hm = BumpHashMap::<usize, usize>::default();

        let n = 20;
        for i in 0..n {
            println!(
                "inserting {}, len={}, cap={}, loaded={}",
                i,
                hm.len(),
                hm.capacity(),
                hm.is_loaded()
            );
            println!("{:?}", hm.spots);
            hm.get_or_default(i, bump);
            for _ in 0..i {
                *hm.get_or_default(i, bump) += 1;
            }
        }

        assert_eq!(hm.len(), n);
        assert_eq!(hm.capacity(), 32);
    }
}
