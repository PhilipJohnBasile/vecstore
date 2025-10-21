fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "server")]
    {
        tonic_prost_build::compile_protos("proto/vecstore.proto")?;
    }

    Ok(())
}
