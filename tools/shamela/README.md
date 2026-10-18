# Copying chosen books out of a Shamela 4 library

Shamela 4 keeps the text of every book in `database/store/page` (a Lucene search index, about 13 GB). The
per-book files in `database/book/` hold only page numbers and headings. `ShamelaExport.java` reads the index
and writes, for each chosen book, small UTF-8 files that Muhawir imports:

- `<book>.page.jsonl`: one line per page, every stored field of the page (its text and its key)
- `<book>.title.jsonl`: the chapter headings
- `<book>.db`: the book's own file (volume and page numbers, the heading tree), copied as it is

Nothing in the library is changed. These files are never put on GitHub (`shamela/` is ignored); they go to the
server only. Only classical books are imported, and editors' footnotes are left out (see SOURCES.md).

## On Windows

1. **Java 21 or newer** (Lucene 10 needs it). `java -version` must say 21 or more; if not, install
   Temurin 21 from adoptium.net (or Zulu 21) and open a new Command Prompt.
2. **Put this file and two Lucene jars in one folder**, e.g. `C:\shamela-export`:
   ```
   curl.exe -O https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/10.4.0/lucene-core-10.4.0.jar
   curl.exe -O https://repo1.maven.org/maven2/org/apache/lucene/lucene-backward-codecs/10.4.0/lucene-backward-codecs-10.4.0.jar
   ```
   If step 3 says the index format is too new, use 10.5.0 in both links instead.
3. **Check the layout** (reads two entries only; also writes `out\inspect.txt`):
   ```
   java -cp "lucene-core-10.4.0.jar;lucene-backward-codecs-10.4.0.jar" ShamelaExport.java "D:\path\to\shamela\database" out --inspect
   ```
4. **Copy the books** (Phase 1: hadith grading, grammar of the Quran, tafsir, everyday practice):
   ```
   java -cp "lucene-core-10.4.0.jar;lucene-backward-codecs-10.4.0.jar" ShamelaExport.java "D:\path\to\shamela\database" out 1263,856,6056,8566,23587,22928,23636,8473,20855,41,2348,1956,197
   ```
   It reads the whole index once (several minutes) and prints how many pages of each book it kept.
5. **Copy the files to the server**, into the folder the database is built from, then rebuild:
   ```
   scp out/*.jsonl out/*.db muhawir:~/muhawir/data/dumps/shamela/
   ssh muhawir "cd ~/muhawir && .venv/bin/python -m muhawir.build_data --from-dir data/dumps"
   ```
   `muhawir/shamela.py` reads them: the author's text only (no editor's footnotes, introduction or indexes),
   each passage placed by heading, volume and page.

| No. | Book | Author (d. AH) |
|---|---|---|
| 1263 | المقاصد الحسنة، ت الغماري | السخاوي (902) |
| 856 | كشف الخفاء، ط القدسي | العجلوني (1162) |
| 6056 | الأسرار المرفوعة في الأخبار الموضوعة | علي القاري (1014) |
| 8566 | المنار المنيف، ت أبي غدة | ابن القيم (751) |
| 23587 | إعراب القرآن | النحاس (338) |
| 22928 | التبيان في إعراب القرآن | العكبري (616) |
| 23636 | المفردات في غريب القرآن | الراغب الأصفهاني (502) |
| 8473 | تفسير ابن كثير، ت السلامة | ابن كثير (774) |
| 20855 | الجامع لأحكام القرآن | القرطبي (671) |
| 41 | تفسير البغوي، ط طيبة | البغوي (516) |
| 2348 | رياض الصالحين، ت الفحل | النووي (676) |
| 1956 | الأذكار، ت الأرنؤوط | النووي (676) |
| 197 | زاد المعاد، ط عطاءات العلم | ابن القيم (751) |

Phase 2 (later): 6910 المغني، 2186 المجموع، 8183 بدائع الصنائع، 1717 الذخيرة، 1673 فتح الباري، 1711 شرح النووي على مسلم.
