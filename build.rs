fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "server")]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(false) // We're building the server, not the client
            .out_dir("src/generated")
            .compile_protos(&["proto/vecstore.proto"], &["proto"])?;
    }

    Ok(())
}
