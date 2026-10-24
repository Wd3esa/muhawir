# Adding a reviewed book

The optional file `data/dumps/approved-extra-sources.json` is read **locally** when the database is built. It is not downloaded, and the text must not be committed to Git. This makes a vetted source easy to add while leaving the existing importers and their citation rules alone. An `author rights reserved` notice is attribution, **not** by itself a grant to copy or publish the text. Record the actual public-domain determination, license, or permission for the particular edition in `rights_basis` and `rights_document`; review it before deployment.

Example schema (the sample sentence is synthetic; do not deploy it as a religious source):

```json
{
  "schema_version": 1,
  "sources": [
    {
      "id": "extra-reviewed-book",
      "name": "اسم الكتاب",
      "about": "المؤلف والمحقق والناشر والطبعة",
      "url": "https://example.org/edition",
      "edition": "طبعة محددة وسنة النشر",
      "rights_holder": "صاحب حقوق هذه الطبعة",
      "rights_basis": "permission",
      "rights_document": "رابط الإذن أو مرجعه وتاريخه",
      "passages": [
        {
          "location": "الجزء 1، الباب 2، ص 3",
          "kind": "other",
          "text": "نص تجريبي مُراجع حرفيًا.",
          "keywords": "عبارات بحث فقط"
        }
      ]
    }
  ]
}
```

Use a stable ASCII ID starting with `extra-`. Allowed rights bases are `public-domain`, `license`, and `permission`. An HTTPS edition URL, named rights holder, edition, rights document, location, and passage text are required. Hadith passages require the source's own nonempty `grade`; a Quran passage must use the canonical Quran importer instead. The importer creates passage IDs `extra-reviewed-book:1`, `:2`, … in file order. Keep text and citation coordinates exact; keep interpretive search words in `keywords` rather than altering the quote.

Append **new books at the end**. Do not reorder, insert into, or edit books already indexed while reusing vectors: a changed text or shifted passage ID requires a full vector rebuild. The manifest hash and metadata enter the database provenance. Validate the whole book with a subject specialist, rights reviewer, and retrieval test; the schema checks alone cannot establish authenticity or legal permission.

Run the fast structural check first, without loading the 94,431 existing passages:

```bash
python -m muhawir.extra_sources data/dumps
```

With the existing dumps available, build to a temporary database first:

```bash
python -m muhawir.build_data --from-dir data/dumps --out data/muhawir-candidate.db
python scripts/eval_retrieval.py --db data/muhawir-candidate.db
```

Check source names, passage count, citation locations, retrieval examples, and SHA-256 provenance. Only after review, replace the production corpus in a controlled deployment, and rebuild or extend the vector index from the **matching** database with `python -m muhawir.vectors build`. An index with missing IDs falls back to keyword search, but an older index can still carry embeddings for changed text with the **same** IDs. Preserve existing passages byte-for-byte when extending; if an earlier passage changes, deliberately rebuild its vectors. Do not call the answer model as a substitute for evaluation. Keep a backup and restore it if the new book reduces answer quality. See [SOURCES.md](../SOURCES.md) for the existing approved-source inventory.
