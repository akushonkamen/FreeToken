from __future__ import annotations

from prometheus.daemon import main  # package dispatch: client verb → client, else → server

raise SystemExit(main(prog="python -m prometheus.daemon"))
