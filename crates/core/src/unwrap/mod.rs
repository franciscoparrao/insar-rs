//! Desenrollado de fase 2D mínimo (alcance MVP): flood-fill guiado por un
//! mapa de calidad (coherencia si está disponible), integrando saltos ±2π
//! entre vecinos. No es un reemplazo de SNAPHU — ver PLAN.md.

use ndarray::{Array2, Array3};

use crate::error::Result;
use crate::types::{IfgStack, UnwrappedStack};

/// Desenrolla un interferograma 2D. `wrapped` en radianes (-π, π].
/// `quality`: mapa opcional (mayor = mejor); si es None se usa calidad
/// uniforme y semilla en el centro de la imagen. NaN se propaga.
pub fn unwrap_2d(wrapped: &Array2<f32>, quality: Option<&Array2<f32>>) -> Result<Array2<f32>> {
    let _ = (wrapped, quality);
    todo!("Fase 2, módulo unwrap — ver PLAN.md")
}

/// Desenrolla cada interferograma del stack (paralelizable por capa).
/// `coherence`: stack opcional con el mismo layout que `stack.data`.
pub fn unwrap_stack(stack: &IfgStack, coherence: Option<&Array3<f32>>) -> Result<UnwrappedStack> {
    let _ = (stack, coherence);
    todo!("Fase 2, módulo unwrap — ver PLAN.md")
}
