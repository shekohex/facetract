use argh::FromArgs;
use facetract::FaceTract;

/// Facetract CLI Tool
/// See How Many Faces in the Picture.
#[derive(FromArgs, PartialEq, Debug)]
struct Cli {
    /// the Picture path
    #[argh(positional)]
    paths: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    // Turn off tensorflow logging.
    std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "3");
    let Cli { paths, .. } = argh::from_env();
    let ft = FaceTract::default();
    for path in paths {
        let img = image::open(&path)?;
        let res = ft.detect(img)?;
        println!("There is {} face(s) in {}", res.len(), path);
    }
    Ok(())
}
