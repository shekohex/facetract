use facetract::FaceTract;
#[test]
fn it_loads_model() { FaceTract::new(); }
#[test]
fn it_works() {
    let img = image::open("tests/images/adel.jpeg").unwrap();
    let ft = FaceTract::default();
    let faces = ft.detect(img).unwrap();
    assert_eq!(faces.len(), 1);
}

#[test]
fn no_cats() {
    let img = image::open("tests/images/cat.jpeg").unwrap();
    let ft = FaceTract::default();
    let faces = ft.detect(img).unwrap();
    assert_eq!(faces.len(), 0);
}
