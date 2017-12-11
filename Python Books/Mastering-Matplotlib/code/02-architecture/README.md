# matplotlib Architecture

This notebook is one in a series. All of them are listed in order here:
 * https://github.com/masteringmatplotlib/notebooks

## Dependencies

The following are required for these examples:

 * Python 3.4.2
 * graphviz header files

For Python, download 3.4 from python.org. For graphviz on Mac OS X:

```
$ brew install graphviz
```

## Instructions

The following will set up a local matplotlib environment for you, and start an
IPython Notebook server:

```bash
  $ make
```

At which point a browser window will open, with a view of this notebook.

Note that this notebook assumes you have the Graphviz include files installed
under ``/usr/local``. If this is not the case, you will want to manually
install pygraphviz into this virtual env.
