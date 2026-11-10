"""``python -m r3con``: the command line, as ``r3con`` is, except in the Mac app, which
Briefcase starts this way: there it opens the app's window (:mod:`r3con.app.window`)."""

from importlib import metadata

if __name__ == "__main__":
    # Only the Mac app has an "r3con" distribution, which Briefcase writes there; pip
    # installs "r3context".
    if any(metadata.distributions(name="r3con")):
        from r3con.app import window

        window.main()
    else:
        from r3con import cli

        raise SystemExit(cli.main())
