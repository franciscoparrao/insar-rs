//! Enganche insar-rs → swarm-abm: el campo de deformación InSAR como **entorno**
//! de un modelo basado en agentes de **exposición**.
//!
//! La velocidad LOS (|v|, cm/año) define un mapa de peligro; agentes-población
//! caminan aleatoriamente sobre el terreno coherente y acumulan exposición al
//! pisar celdas de alta deformación. Es el eslabón deformación → impacto (DRR):
//! insar-rs dice *dónde y cuánto* se mueve el suelo; swarm-abm simula *a quién
//! afecta*. Mismo patrón de los engines hermanos (geostat, Smelt): acoplamiento
//! suelto vía dev-dependency.
//!
//! Uso: cargo run --release -p insar-core --example exposure_abm -- <export_dir>

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use swarm_core::prelude::*;

const HAZARD_THR: f32 = 5.0; // cm/año: umbral de "deformación significativa"

#[derive(Deserialize)]
struct Meta { rows: usize, cols: usize }

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let mut b = Vec::new();
    fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    assert_eq!(b.len(), n * 4);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Person {
    pos: Pos,
}

struct World {
    agents: AgentSet<Person>,
    hazard: Grid2D<f32>,    // |velocidad LOS| (cm/año)
    walkable: Grid2D<bool>, // celdas coherentes (donde puede haber población)
    exposure: Grid2D<u32>,  // veces que un agente pisó esta celda en peligro
    n_agents: usize,
    exposed_now: usize,     // agentes sobre celda de peligro en el paso actual
}

impl Agent for Person {
    type Model = World;
    fn step(&mut self, _id: AgentId, w: &mut World, rng: &mut SimRng) {
        // Camina a un vecino caminable (random walk restringido al terreno).
        if let Some(dest) = w.hazard.random_neighbor(self.pos, Neighborhood::Moore, rng)
            && w.walkable[dest]
        {
            self.pos = dest;
        }
        if w.hazard[self.pos] > HAZARD_THR {
            w.exposure[self.pos] += 1;
            w.exposed_now += 1;
        }
    }
}

impl Model for World {
    type Agent = Person;
    fn agents(&self) -> &AgentSet<Person> { &self.agents }
    fn agents_mut(&mut self) -> &mut AgentSet<Person> { &mut self.agents }
    fn before_step(&mut self, _rng: &mut SimRng) { self.exposed_now = 0; }
}

fn main() {
    let dir = std::env::args().nth(1).expect("export_dir");
    let dir = Path::new(&dir);
    let m: Meta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let (nr, nc) = (m.rows, m.cols);
    let vel = read_f32(&dir.join("velocity.f32"), nr * nc);
    let tcoh = read_f32(&dir.join("tcoh.f32"), nr * nc);

    // Entorno: peligro = |velocidad| cm/año; caminable = coherente.
    let hazard = Grid2D::from_fn(nc, nr, |p: Pos| {
        let i = p.y * nc + p.x;
        if vel[i].is_finite() { (vel[i] * 100.0).abs() } else { 0.0 }
    });
    let walkable = Grid2D::from_fn(nc, nr, |p: Pos| {
        let i = p.y * nc + p.x;
        tcoh[i].is_finite() && tcoh[i] > 0.5 && vel[i].is_finite()
    });

    // Población: ~4000 agentes repartidos en celdas caminables (determinista).
    let walk_cells: Vec<Pos> = (0..nr)
        .flat_map(|r| (0..nc).map(move |c| Pos::new(c, r)))
        .filter(|&p| walkable[p])
        .collect();
    let target = 4000;
    let stride = (walk_cells.len() / target).max(1);
    let mut agents = AgentSet::new();
    for p in walk_cells.iter().step_by(stride) {
        agents.insert(Person { pos: *p });
    }
    let n_agents = agents.len();
    let haz_frac = walk_cells.iter().filter(|&&p| hazard[p] > HAZARD_THR).count() as f64
        / walk_cells.len() as f64;
    println!(
        "entorno {}×{}: {} celdas caminables ({:.1}% en peligro >{} cm/año); {} agentes",
        nc, nr, walk_cells.len(), 100.0 * haz_frac, HAZARD_THR, n_agents
    );

    let world = World {
        agents,
        hazard,
        walkable,
        exposure: Grid2D::new(nc, nr),
        n_agents,
        exposed_now: 0,
    };

    let mut sim = Simulation::new(world, 42);
    sim.add_reporter("exposed_frac", |w: &World| w.exposed_now as f64 / w.n_agents.max(1) as f64);
    let steps = 300;
    sim.run(steps);

    // Serie de fracción expuesta + mapa de exposición acumulada.
    if let Some(s) = sim.data().series("exposed_frac") {
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let last = *s.last().unwrap();
        println!("fracción de población en peligro: media {:.3}, final {:.3} (sobre {steps} pasos)", mean, last);
    }
    let exposure = &sim.model.exposure;
    let mut total = 0u64;
    let mut hit_cells = 0usize;
    let mut out = vec![0f32; nr * nc];
    for r in 0..nr {
        for c in 0..nc {
            let e = exposure[Pos::new(c, r)];
            out[r * nc + c] = e as f32;
            total += e as u64;
            if e > 0 { hit_cells += 1; }
        }
    }
    let mut b = Vec::with_capacity(out.len() * 4);
    for &v in &out { b.extend_from_slice(&v.to_le_bytes()); }
    fs::write(dir.join("exposure.f32"), b).unwrap();
    println!(
        "exposición acumulada: {} pisadas-en-peligro sobre {} celdas → exposure.f32",
        total, hit_cells
    );
}
