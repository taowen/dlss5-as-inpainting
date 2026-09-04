# Example image sources

The downloaded inputs are used only as reproducible evaluation fixtures. Each
source page was checked before download; the original files and their SHA-256
values are recorded in `examples/cases/native/manifest.json`.

| local file | source page | stated reuse status |
|---|---|---|
| `input/blue_marble.jpg` | [Nasa blue marble](https://commons.wikimedia.org/wiki/File:Nasa_blue_marble.jpg) | Public domain NASA/USGS/NOAA image |
| `input/scenic_landscape.jpg` | [Vintage scenic landscape photo](https://commons.wikimedia.org/wiki/File:Vintage_scenic_landscape_photo.jpg) | Public domain; U.S. Fish and Wildlife Service |
| `input/stone_texture.jpg` | [STONE TEXTURE](https://commons.wikimedia.org/wiki/File:STONE_TEXTURE.jpg) | CC0 |
| `input/portrait_cc0.jpg` | [Portrait of a Man, MET DP326401](https://commons.wikimedia.org/wiki/File:-Portrait_of_a_Man-_MET_DP326401.jpg) | CC0 / The Metropolitan Museum of Art Open Access |

The normalized 256x256 images under `normalized/` are generated from these
files by the experiment tool. The synthetic depth maps are not downloaded
assets: they are controlled `1 - linear_luma` proxies used to isolate depth
sensitivity.
