# CDE parser

Currently only `strip_html.py`. 

## Use
Use requires some preprocessing of the JSON. The parser is based on `pydantic` class modeling of the data model in the [CDE API](https://cde.nlm.nih.gov/api). There are elements (pydantic `Fields`)
with names that have leading underscores. This cause `pydantic` to struggle. Therefore one has to first alter the names to add a leading underscore. However, some data values have patterns that are similar 
and a standard `regex` does not work. It is also necessery to `pretty`fy the JSON so that the field names are preceded by whitespace. Thus use is:  

1. prettyfy JSON
2. sed substitute `sed -i 's/ "\(__*[iv]\)/ "x\1/' <file>`. Beware of `-i` option, it overwrites in place.
3. `python strip_html.py <cleaned_file>`
4. sed substitute `sed -i 's/ x"\(__*[iv]\)/ "\1/' <file>`

