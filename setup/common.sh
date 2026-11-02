# Sets `here` to the root of this repo and sources the
# `.env` file for the specified site.

here=$(cd "$(dirname "$0")" && pwd)
source "$here/sites/$1.env"
