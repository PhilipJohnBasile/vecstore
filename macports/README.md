# MacPorts candidate

The candidate pins VecStore 0.1.0 source `1868a6273174f58856af25ba6c45eb057d3bbfd0`, with verified archive checksums and all 573 registry crates from that revision's Cargo.lock. It fixes the reviewer-requested pkg-config path dependency and GitHub maintainer syntax. The Cargo PortGroup's Rust dependencies are retained with `depends_build-append`, and the install path uses its current target-triplet API.

`port lint`, a sandboxed offline MacPorts build and installed `vecstore-server --help` remain required. MacPorts is not installed on the validation host. The old upstream PR #29710 is closed and has not been reopened.

References: [Cargo PortGroup](https://github.com/macports/macports-ports/blob/master/_resources/port1.0/group/cargo-1.0.tcl), [Cargo dependency manifest](https://github.com/macports/macports-ports/blob/master/_resources/port1.0/group/cargo_fetch-1.0.tcl), [maintainer syntax](https://guide.macports.org/chunked/reference.html#reference.keywords).
