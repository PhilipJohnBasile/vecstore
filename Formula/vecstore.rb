class Vecstore < Formula
  desc "High-performance vector database with RAG toolkit"
  homepage "https://github.com/PhilipJohnBasile/vecstore"
  url "https://github.com/PhilipJohnBasile/vecstore/archive/refs/tags/v0.0.1.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256"
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
