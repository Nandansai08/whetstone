# Examples

This section showcases example builds using Whetstone.

## Example 1: Building a standard Python Module

Run:
```bash
whetstone "build a function add(a, b) that returns the sum of two integers"
```

Whetstone will:
1. Decompose task into planning stages.
2. Compile and critique.
3. Test using `pytest` inside a subprocess.
4. Export the clean python script.

## Example 2: Building a database Schema

Run:
```bash
whetstone "build a sqlite schema representing users and roles with a many-to-many relationship"
```

Whetstone will:
1. Decompose the request into SQL statements.
2. Run database migration tests inside the SQLite memory verifier sandbox.
3. Grade output using cross-model SQL rubrics.
