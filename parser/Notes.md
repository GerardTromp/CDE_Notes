# Required changes

## Parsing forms  
1. All records are parsed, but not all fields are written to the output file.
2. Forms contain html tables. This requires the HTML cleaner to detect table tags and then perform a table-specific conversion to JSON. One problem then is that the model will need to permit the relevant key to contain a dict, and not just a string.
