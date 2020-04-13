#![deny(
    unsafe_code,
    missing_debug_implementations,
    missing_copy_implementations,
    elided_lifetimes_in_paths,
    rust_2018_idioms,
    clippy::fallible_impl_from,
    clippy::missing_const_for_fn
)]

use image::{GenericImageView, Pixel};
use num_traits::cast::NumCast;
use tensorflow::{
    Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs,
    Tensor,
};

/// The Pretrained model
/// downloaded from: https://github.com/blaueck/tf-mtcnn
static MODEL: &[u8] = include_bytes!("../assets/mtcnn.pb");

/// Holds two points (x1, y1) and (x2, y2) that represent a `Bordered Box`
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct FaceLocationBox {
    /// The X-axis part of the first point
    pub x1: f32,
    /// The X-axis part of the second point
    pub x2: f32,
    /// The Y-axis part of the first point
    pub y1: f32,
    /// The Y-axis part of the second point
    pub y2: f32,
}

impl From<[f32; 4]> for FaceLocationBox {
    fn from(c: [f32; 4]) -> Self {
        Self {
            x1: c[0],
            x2: c[1],
            y1: c[2],
            y2: c[3],
        }
    }
}

impl FaceLocationBox {
    /// Calculates the Width of the `Bordered Box`
    pub fn width(&self) -> u32 { (self.x2 - self.x1) as u32 }

    /// Calculates the Height of the `Bordered Box`
    pub fn height(&self) -> u32 { (self.y2 - self.y1) as u32 }
}

/// Holds the location of the extracted Face, also  how likely it is that face
/// of a Human!
#[derive(Copy, Clone, Debug)]
pub struct Face {
    location: FaceLocationBox,
    prob: f32,
}

/// The Errors always happens :)
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Tensorflow Error: {}", 0)]
    TFError(#[from] tensorflow::Status),
}

impl Face {
    /// Get the Location of that Face
    pub const fn location_box(&self) -> &FaceLocationBox { &self.location }

    /// How likely it is a Human Face
    pub const fn probability(&self) -> f32 { self.prob }
}

#[derive(Debug)]
pub struct FaceTract {
    graph: Graph,
    min_size: f32,
    factor: f32,
    thresholds: [f32; 3],
}

impl Default for FaceTract {
    fn default() -> Self { Self::new(40.0, 0.709, [0.6, 0.7, 0.7]) }
}

impl FaceTract {
    pub fn new(min_size: f32, factor: f32, thresholds: [f32; 3]) -> Self {
        let mut graph = Graph::new();
        graph
            .import_graph_def(MODEL, &ImportGraphDefOptions::new())
            .expect("bad model loaded!");
        Self {
            graph,
            min_size,
            factor,
            thresholds,
        }
    }

    /// Set the Current `factor` arg
    pub const fn set_factor(mut self, factor: f32) -> Self {
        self.factor = factor;
        self
    }

    /// Set the Current `min_size` arg
    pub const fn set_min_size(mut self, min_size: f32) -> Self {
        self.min_size = min_size;
        self
    }

    /// Set the Current `thresholds` arg
    pub const fn set_thresholds(mut self, thresholds: [f32; 3]) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Detect Faces in an Image
    pub fn detect(
        &self,
        img: impl GenericImageView,
    ) -> Result<Vec<Face>, Error> {
        let session = Session::new(&SessionOptions::new(), &self.graph)?;
        let flattened: Vec<f32> = img
            .pixels()
            .map(|(_x, _y, pixel)| pixel)
            .map(|p| {
                let c = p.channels();
                vec![
                    NumCast::from(c[2]).unwrap(),
                    NumCast::from(c[1]).unwrap(),
                    NumCast::from(c[0]).unwrap(),
                ]
            })
            .flatten()
            .collect();
        let input = Tensor::new(&[img.height() as u64, img.width() as u64, 3])
            .with_values(&flattened)?;
        let min_size = Tensor::new(&[]).with_values(&[self.min_size])?;
        let thresholds = Tensor::new(&[3]).with_values(&self.thresholds)?;
        let factor = Tensor::new(&[]).with_values(&[self.factor])?;
        let mut args = SessionRunArgs::new();
        args.add_feed(
            &self.graph.operation_by_name_required("min_size")?,
            0,
            &min_size,
        );
        args.add_feed(
            &self.graph.operation_by_name_required("thresholds")?,
            0,
            &thresholds,
        );
        args.add_feed(
            &self.graph.operation_by_name_required("factor")?,
            0,
            &factor,
        );
        args.add_feed(
            &self.graph.operation_by_name_required("input")?,
            0,
            &input,
        );

        let bbox = args
            .request_fetch(&self.graph.operation_by_name_required("box")?, 0);
        let prob = args
            .request_fetch(&self.graph.operation_by_name_required("prob")?, 0);
        session.run(&mut args)?;
        let bbox_res: Tensor<f32> = args.fetch(bbox)?;
        let prob_res: Tensor<f32> = args.fetch(prob)?;

        let faces: Vec<_> = bbox_res
            .chunks_exact(4)
            .zip(prob_res.iter())
            .map(|(bbox, &prob)| Face {
                location: FaceLocationBox {
                    y1: bbox[0],
                    x1: bbox[1],
                    y2: bbox[2],
                    x2: bbox[3],
                },
                prob,
            })
            .collect();
        Ok(faces)
    }
}
