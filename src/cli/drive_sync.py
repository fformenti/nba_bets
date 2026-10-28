"""Sync selected repo paths with My Drive/nba_bets/ for Colab.

    make drive-auth                       # once: browser consent, stores the token
    make drive-ls DRIVE_PATH=build        # inspect a Drive folder
    make drive-sync DRY_RUN=1             # show what would change
    make drive-sync ONLY=outputs          # pull outputs/runs and outputs/comparisons only

Code and config are push-only mirrors; outputs/ is pulled from Drive, never
uploaded or deleted, figures skipped. See src/gdrive/sync.py and README.md ("Google Drive sync").
"""

import argparse

from src.config.paths import PROJECT_ROOT
from src.gdrive import drive, sync
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the repo with Google Drive")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("auth", help="Run the OAuth browser login and store the token")
    ls = commands.add_parser("ls", help="List a folder under the Drive root")
    ls.add_argument("path", nargs="?", default="", help="Path under My Drive/nba_bets/")
    ls.add_argument("--recursive", "-r", action="store_true")
    run = commands.add_parser("sync", help="Push code/config, pull outputs/")
    run.add_argument("--dry-run", action="store_true", help="Print the plan, change nothing")
    run.add_argument("--only", choices=sorted(sync.SCOPE_GROUPS), help="Sync one group")
    args = parser.parse_args()
    setup_logging()

    if args.command == "auth":
        print(f"Token saved to {drive.run_login()}")  # noqa: T201
        return

    service = drive.get_service()
    root = sync.find_drive_root(service)
    if args.command == "ls":
        print(sync.format_listing(service, root.id, args.path, args.recursive))  # noqa: T201
        return

    actions = sync.plan_sync(service, PROJECT_ROOT, root.id, args.only)
    print(sync.format_plan(actions))  # noqa: T201
    if args.dry_run:
        print("\nDry run: nothing was changed.")  # noqa: T201
        return
    sync.apply_actions(service, actions, PROJECT_ROOT, root.id)
    print("\nSync complete.")  # noqa: T201


if __name__ == "__main__":
    main()
