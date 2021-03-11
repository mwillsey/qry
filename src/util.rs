use std::hash::{BuildHasherDefault, Hash, Hasher};

use bumpalo::Bump;

// macro_rules! probe {
//     ($self:ident, $k:expr) => {{
//         probe!($self, $k, get_unchecked)
//     }};
//     (mut $self:ident, $k:expr) => {{
//         probe!($self, $k, get_unchecked_mut)
//     }};
//     ($self:ident, $k:expr, $get:ident) => {{
//         let cap = $self.capacity();
//         let mut i = $self.hash($k) as usize % cap;
//         loop {
//             match unsafe { $self.spots.$get(i) } {
//                 Some((k2, v)) if $k == k2 => break Ok(v),
//                 Some(_) => {
//                     i += 1;
//                     if i == cap {
//                         i = 0;
//                     }
//                 }
//                 spot => break Err(spot),
//             }
//         }
//     }};
// }

#[allow(dead_code)]
pub struct BumpHashMap<'bump, K: 'bump, V: 'bump, S = BuildHasherDefault<rustc_hash::FxHasher>> {
    len: usize,
    spots: &'bump mut [Option<(K, V)>],
    builder: S,
}

impl<'bump, K: 'bump, V: 'bump> Default for BumpHashMap<'bump, K, V> {
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
    K: 'bump + Hash + Eq,
    V: 'bump,
    S: std::hash::BuildHasher,
{
    const GROWTH_FACTOR: usize = 2;
    const MIN_CAPACITY: usize = 4;

    pub fn hash(&self, k: &K) -> u64 {
        let mut hasher = self.builder.build_hasher();
        k.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let cap = self.capacity();
        if cap == 0 {
            return None
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
                _ => return None
            }
        }
    }

    pub fn get_or_default(&mut self, k: K, bump: &'bump Bump) -> &mut V
        where V: Default
    {
        if self.capacity() == 0 {
            self.grow(bump);
        }

        let cap = self.capacity();
        let mut i = self.hash(&k) as usize % cap;
        let ptr = self.spots.as_mut_ptr();
        loop {
            let r = unsafe {&mut *(ptr.offset(i as isize)) };
            match r {
                Some((k2, v)) if &k == k2 => return v,
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                _ => {
                    let spot = if !self.is_loaded() {
                        i
                    } else {
                        self.grow(bump);
                        match self.probe_mut(&k) {
                            Ok(_) => unreachable!(),
                            Err(i) => i,
                        }
                    };
                    self.len += 1;

                    let spot = unsafe { self.spots.get_unchecked_mut(i) };
                    *spot = Some((k, V::default()));
                    return &mut spot.as_mut().unwrap().1
                }
            }
        }

        // let spot = match self.probe_mut(&k) {
        //     Ok(v) => return v,
        //     Err(spot) => spot,
        // };

        todo!()
                // let spot = if !self.is_loaded() {
                //     // unsafe { &mut *(spot as *mut _) }
                // } else {
                //     self.grow(bump);
                //     match self.probe_mut(&k) {
                //         Ok(_) => unreachable!(),
                //         Err(spot) => spot,
                //     }
                // };

                // *spot = Some((k, V::default()));
                // &mut spot.as_mut().unwrap().1
        //     }
        // };
    }


    // fn probe_mut(&mut self, k: &K) -> Result<&mut V, &mut Option<(K, V)>> {
    //     match self.probe_impl(k) {
    //         Ok(i) => unsafe {
    //             let spot = self.spots.get_unchecked_mut(i);
    //             Ok(&mut spot.as_mut().unwrap_or_else(|| std::hint::unreachable_unchecked()).1)
    //         }
    //         Err(i) => Err(unsafe { self.spots.get_unchecked_mut(i) })
    //     }
    // }

    // fn probe(&self, k: &K) -> Result<&V, &Option<(K, V)>> {
    //     match self.probe_impl(k) {
    //         Ok(i) => unsafe {
    //             let spot = self.spots.get_unchecked(i);
    //             Ok(&spot.as_ref().unwrap_or_else(|| std::hint::unreachable_unchecked()).1)
    //         }
    //         Err(i) => Err(unsafe { self.spots.get_unchecked(i) })
    //     }
    // }

    fn probe_mut(&mut self, k: &K) -> Result<&mut V, usize> {
        let cap = self.capacity();
        let mut i = self.hash(k) as usize % cap;
        let ptr = self.spots.as_mut_ptr();
        loop {
            let r = unsafe {&mut *(ptr.offset(i as isize)) };
            match r {
                Some((k2, v)) if k == k2 => return Ok(v),
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                _ => return Err(i),
            }
        }
    }

    fn probe(&self, k: &K) -> Result<&V, usize> {
        let cap = self.capacity();
        let mut i = self.hash(k) as usize % cap;
        loop {
            match unsafe { self.spots.get_unchecked(i) } {
                Some((k2, v)) if k == k2 => return Ok(v),
                Some(_) => {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                _ => return Err(i),
            }
        }
    }

    // fn insert_impl(&mut self, k: K, v: V) {
    //     debug_assert!(self.load_factor() < Self::MAX_LOAD);
    //     match probe!(mut self, &k) {
    //     }
    // }

    fn grow(&mut self, bump: &'bump Bump) {
        // debug_assert!(self.capacity() == 0 || self.load_factor() > Self::MAX_LOAD);
        let new_cap = match self.capacity() {
            0 => Self::MIN_CAPACITY,
            cap => cap * Self::GROWTH_FACTOR,
        };
        println!("growing to {}", new_cap);
        let new_spots = bump.alloc_slice_fill_default(new_cap);
        let old_spots = std::mem::replace(&mut self.spots, new_spots);
        for spot in old_spots.iter_mut() {
            if let Some((k, v)) = spot.take() {
                match self.probe_mut(&k) {
                    Ok(_) => unreachable!(),
                    Err(new_spot) => {
                        unsafe {*self.spots.get_unchecked_mut(new_spot) = Some((k, v)) }
                        // *new_spot = Some((k, v))
                    }
                };
            }
        }
        println!("grown to {}", new_cap);
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
            println!("inserting {}, len={}, cap={}, loaded={}", i, hm.len(), hm.capacity(), hm.is_loaded());
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
