# CDE observations  

The project uses the data on the CDE repository. There is an extensive API that defines the Form and Cd models. 

## Data-Model disagreement  

### Elements defined in `Form`  are found in `Cd` 
Example: 
`formElements` defined in `Form` are in `Cd`, see `ctyKfVm8k_B`

### Data contain embedded HTML  

### Missing model keys (Fields)  
#### cdeTinyIds  
The data contain a list element at the highest level in `Form`, but this is not in the API documentation.

## Data contain disclaimer text 
Disclaimer text, important as it may be, needs to be removed prior to use. 

## Python `pydantic` and keys (Fields) with leading underscores  
Pydantic simply is not able to parse these fields which are treated as private correctly. 
The appropriate `pydantic` approach is to give the field an alias, **BUT** this fails
when the field is also optional, requiring the keyword `None`.   
Fields must be preprocess to add an alphabetical prefix. 
Can be done with `sed`, but requires a pass before and after.
