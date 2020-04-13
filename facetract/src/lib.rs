use image::{GenericImageView, Pixel};
use num_traits::cast::NumCast;
use tensorflow::{
    Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs,
    Tensor,
};
static MODEL: &[u8] = include_bytes!("../assets/mtcnn.pb");

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct FaceLocationBox {
    pub x1: f32,
    pub x2: f32,
    pub y1: f32,
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
    pub fn width(&self) -> u32 { (self.x2 - self.x1) as u32 }

    pub fn height(&self) -> u32 { (self.y2 - self.y1) as u32 }
}

#[derive(Copy, Clone, Debug)]
pub struct Face {
    location: FaceLocationBox,
    prob: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Tensorflow Error: {}", 0)]
    TFError(#[from] tensorflow::Status),
}

impl Face {
    pub fn location_box(&self) -> &FaceLocationBox { &self.location }

    pub fn probability(&self) -> f32 { self.prob }
}

#[derive(Debug)]
pub struct FaceTract {
    graph: Graph,
}

impl Default for FaceTract {
    fn default() -> Self { Self::new() }
}

impl FaceTract {
    pub fn new() -> Self {
        let mut graph = Graph::new();
        graph
            .import_graph_def(MODEL, &ImportGraphDefOptions::new())
            .expect("bad model loaded!");
        Self { graph }
    }

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
        let min_size = Tensor::new(&[]).with_values(&[40f32])?;
        let thresholds =
            Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
        let factor = Tensor::new(&[]).with_values(&[0.709f32])?;
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
