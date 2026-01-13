# Introduction to Notebooks in XNAT

This notebook provides an introduction to working with Jupyter Notebooks within the XNAT environment. It covers essential tools, data exploration, and programmatic access to XNAT resources.

## Table of Contents

1. [Basic Commands & Tools](#1-basic-commands--tools)
2. [Finding & Exploring Mounted Data](#2-finding--exploring-mounted-data)
3. [Using XNATpy SDK](#3-using-xnatpy-sdk)
4. [Putting it All Together](#4-putting-it-all-together)

---

## 1. Basic Commands & Tools

### Shell/Terminal Fundamentals
- Navigation: `pwd`, `ls`, `cd`
- File operations: `cp`, `mv`, `mkdir`, `rm`
- Viewing files: `cat`, `head`, `tail`, `less`
- Searching: `find`, `grep`
- Package management: `pip install`, `pip list`

### Jupyter Notebook Essentials
- Running shell commands with `!` prefix
- Magic commands (`%` and `%%`) - especially `%pwd`, `%ls`, `%env`
- Accessing documentation with `?` and `??`
- Tab completion for discovery
- Keyboard shortcuts (run cell, interrupt kernel, restart)

### Python Standard Library Tools
- `os` and `pathlib` for file system operations
- `sys` for system-level info
- `glob` for file pattern matching
- Environment variables with `os.environ`

### Common Data Libraries
- `pandas` for tabular data
- Basic visualization with `matplotlib`

---

## 2. Finding & Exploring Mounted Data

### 2.1 Locating the /data Directory
- Using shell commands (`ls`, `find`, `pwd`) to locate mounted data
- Navigating to `/data` from the notebook environment
- Using Python's `os` and `pathlib` to programmatically find data paths

### 2.2 Exploring the Data Directory Structure
- Listing directory contents with `ls` and `os.listdir()`
- Understanding the XNAT data hierarchy (projects, subjects, experiments)
- Using `glob` patterns to discover files
- Inspecting file types and sizes

### 2.3 Loading Data into Pandas
- Reading CSV/TSV files with `pd.read_csv()`
- Loading JSON data with `pd.read_json()`
- Basic DataFrame inspection (`head()`, `info()`, `describe()`)
- Handling common data loading issues

---

## 3. Using XNATpy SDK

### 3.1 Installation & Setup
- Checking if XNATpy is installed (`pip list`, `import xnat`)
- Installing XNATpy if needed (`pip install xnat`)
- Verifying successful installation

### 3.2 Authenticating with XNAT
- Creating a connection to the XNAT server
- Using credentials and environment variables
- Managing sessions and connection context

### 3.3 Finding Projects & Data
- Listing available projects
- Navigating subjects and experiments
- Querying for specific data types
- Understanding XNAT resource hierarchy

### 3.4 Working with Mounted Data via XNATpy
- Using `.data_dir` to get mounted paths for XNATpy objects
- Combining API queries with mounted file access
- Best practices for data access

---

## 4. Putting it All Together

### 4.1 Creating a Pandas DataFrame
- Iterating through XNATpy hierarchy to collect metadata
- Building structured records from projects, experiments, and scans
- Converting to pandas DataFrame
- Adding file paths using mounted data directory

### 4.2 Filtering the Data
- *Coming soon...*

---
