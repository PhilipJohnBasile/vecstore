# Packaging status

The Nix and Conda candidates describe VecStore **0.1.0** from immutable source `1868a6273174f58856af25ba6c45eb057d3bbfd0`, not a v1.0.0 release. Local copies now match the corrected upstream submissions. The source archive SHA-256 was independently verified as `09b19547e24e93ca1fe704485df1c4fa25464de635d617bae9eb7e4310596721` on September 10, 2026.

| Candidate | Current evidence | Remaining gate |
| --- | --- | --- |
| Nix / #454087 | Source and cargo hashes, finalAttrs, update script and maintainer entry present in the PR; Linux builds green | Maintainer reconciliation and completed Darwin builds; local Nix is unavailable |
| Conda / #31254 | Corrected compiler/stdlib dependencies, license bundle, quoted install prefix; hosted checks green | Maintainer review; Windows is intentionally skipped |
| MacPorts / closed #29710 | Source checksums, all locked crate checksums, pkg-config dependency, maintainer syntax and Cargo install path prepared | Real `port lint`, offline build and installed executable smoke; MacPorts is unavailable locally |

The Nix expression is intended for nixpkgs with the maintainer entry carried by #454087. These candidates have not been published or accepted by their package managers. Old upstream PR descriptions still refer to v1.0.0 and should be corrected by the contributor when presenting the candidates.

Other directories (`Formula/`, `aur/`, `scoop/`, `snap/`, `chocolatey/`, `winget/`) retain their prior template status. Their artifacts and installation paths have not been qualified by this work.

Keep the recipe source revision, version and all dependency checksums together. Replacing an unpacked Nix source hash with the compressed tarball digest, or retaining an old Cargo hash after changing source, produces an invalid package. Release automation must reject mismatched recipe pins before any external submission.
