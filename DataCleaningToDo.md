# To Do

### 2025-06-24 Tasks for CDE project

1. Remove HTML **Done --mostly** (still need to detect html tables and return json)
2. Identify history, use only recent
3. Identify repeated phrases (8+ words ignoring case), export list first for review. EAV format, phrase plus list of id's
  - Remove selected phrases
4. Identify CDEs with empty fields, especially our fields of interest (Name, Question, Description. Permissible Values)
5. Identify corresponding Forms that have embedded CDEs, i.e., generate Form TinyId: List(CDE TinyIDs) (Possibly reverse for faster iteration)
6. Identify CDE-Forms with CDEs that have permissible values with non-null, non-empty valueNameMeaning (?)
7. Transfer sections (`Description`, `PermissibleValues`, etc.) section(s) (elements) from Form to CDE where the CDE sections (e.g., `PermissibleValues`) are essentially stubs (*dummy*) or empty.

#### Notes

**Execute in order**:

- Remove HTML
- Transfer Form CDE elements to CDEs
- Remove repeated text, e.g. Executive order blurb

### Outcomes

1. The JSON data set cleaned of HTML can be provided back to Matt to share with the CDE maintainers
2. The JSON data set, cleaned of HTML, with complete CDEs transferred from Forms to CDEs (matching TinyID), can be given to Matt.
3. The JSON data set , cleaned of HTML and repeated phrases, and with augmented CDEs, will make up the reference data.
