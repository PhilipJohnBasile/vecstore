class Vecstore < Formula
  desc "High-performance vector database with RAG toolkit"
  homepage "https://github.com/PhilipJohnBasile/vecstore"
  url "https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "96bce955be4d16ccd29e6dbd8bbfe87c1da869b4c76607176ad7ca9c54895448"
  license "MIT"

  depends_on "rust" => :build
  depends_on "protobuf" => :build

  def install
    # Build the server binary (CLI currently has compilation issues)
    system "cargo", "install", "--locked", "--features", "server",
                    "--bin", "vecstore-server", "--path", ".", "--root", prefix
  end

  def caveats
    <<~EOS
      VecStore server has been installed.

      To start the server:
        vecstore-server --db-path ./data/vectors.db

      For library usage in Rust projects:
        cargo add vecstore

      For Python bindings:
        pip install vecstore-rs

      For JavaScript/WASM:
        npm install vecstore-wasm

      Documentation: https://docs.rs/vecstore
    EOS
  end

  test do
    system "#{bin}/vecstore-server", "--help"
  end
end
