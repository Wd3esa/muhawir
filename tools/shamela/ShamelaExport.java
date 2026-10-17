// Copies the text of chosen books out of a Shamela 4 library (database/store/page and store/title, Lucene indexes)
// into small JSON-lines files, one per book, for Muhawir's import step (muhawir/shamela.py). Nothing is changed in the
// library. Only stored text is read; the 13 GB index is never uploaded.
//
// Needs Java 21 and two Lucene 10 jars next to this file (see tools/shamela/README.md):
//   java -cp "lucene-core-10.4.0.jar;lucene-backward-codecs-10.4.0.jar" ShamelaExport.java <database folder> <out folder> 1263,856,...
//   (on Linux or macOS write ":" instead of ";")
// With --inspect instead of book numbers it only prints the field names and two sample entries, to check the layout.
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;

public class ShamelaExport {
    static final Pattern KEY = Pattern.compile("^(\\d+)[-_:/ .](\\d+)$");
    static final String[] BOOK_FIELDS = {"book_id", "book", "bookid", "book_key", "bk", "b"};

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: ShamelaExport <database folder> <out folder> <book numbers, comma separated | --inspect>");
            System.exit(2);
        }
        Path db = Paths.get(args[0]);
        Path out = Paths.get(args[1]);
        Files.createDirectories(out);
        if (args[2].equals("--inspect")) {  // also written to inspect.txt in UTF-8: a Windows console may not show Arabic
            PrintStream file = new PrintStream(Files.newOutputStream(out.resolve("inspect.txt")), true, StandardCharsets.UTF_8);
            PrintStream console = System.out;
            System.setOut(new PrintStream(new OutputStream() {
                public void write(int b) { console.write(b); file.write(b); }
                public void write(byte[] b, int o, int n) { console.write(b, o, n); file.write(b, o, n); }
            }, true, StandardCharsets.UTF_8));
            for (String index : new String[] {"page", "title"}) inspect(db.resolve("store").resolve(index));
            file.close();
            return;
        }
        Set<String> wanted = new LinkedHashSet<>(Arrays.asList(args[2].split("\\s*,\\s*")));
        for (String index : new String[] {"page", "title"}) export(db.resolve("store").resolve(index), out, wanted, index);
        copyBookFiles(db.resolve("book"), out, wanted);
        System.out.println("done: " + out.toAbsolutePath());
    }

    static void inspect(Path index) throws IOException {
        System.out.println("== " + index);
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(index))) {
            System.out.println("entries: " + reader.numDocs());
            for (FieldInfo fi : FieldInfos.getMergedFieldInfos(reader))
                System.out.println("field: " + fi.name + " indexed=" + (fi.getIndexOptions() != IndexOptions.NONE)
                        + " docValues=" + fi.getDocValuesType());
            int shown = 0;
            for (LeafReaderContext leaf : reader.leaves()) {
                StoredFields stored = leaf.reader().storedFields();
                for (int i = 0; i < leaf.reader().maxDoc() && shown < 2; i++, shown++) {
                    Document doc = stored.document(i);
                    System.out.println("-- sample " + (shown + 1));
                    for (IndexableField f : doc.getFields()) System.out.println("  " + f.name() + " = " + shorten(value(f), 160));
                }
                if (shown >= 2) break;
            }
        }
    }

    static void export(Path index, Path out, Set<String> wanted, String kind) throws IOException {
        Map<String, Writer> files = new HashMap<>();
        long seen = 0, kept = 0;
        String rule = null;
        try (DirectoryReader reader = DirectoryReader.open(FSDirectory.open(index))) {
            long total = reader.numDocs();
            for (LeafReaderContext leaf : reader.leaves()) {
                LeafReader r = leaf.reader();
                Bits live = r.getLiveDocs();
                StoredFields stored = r.storedFields();
                for (int i = 0; i < r.maxDoc(); i++) {
                    if (live != null && !live.get(i)) continue;
                    Document doc = stored.document(i);
                    seen++;
                    String[] found = bookOf(doc);
                    if (found == null) {
                        if (seen <= 1000) continue;
                        if (rule == null) {
                            System.err.println("Could not tell which book an entry of " + index + " belongs to. Run with --inspect and send the output.");
                            System.exit(1);
                        }
                        continue;
                    }
                    if (rule == null) { rule = found[1]; System.out.println(kind + ": book number taken from " + rule); }
                    if (!wanted.contains(found[0])) continue;
                    Writer w = files.computeIfAbsent(found[0], b -> open(out.resolve(b + "." + kind + ".jsonl")));
                    w.write(json(doc));
                    w.write('\n');
                    kept++;
                    if (seen % 200000 == 0) System.out.printf("%s: %,d of %,d read, %,d kept%n", kind, seen, total, kept);
                }
            }
        } finally {
            for (Writer w : files.values()) w.close();
        }
        System.out.printf("%s: %,d read, %,d kept, books found: %s%n", kind, seen, kept, new TreeSet<>(files.keySet()));
    }

    // the book an entry belongs to, and the rule that told it: a field naming the book, or a key «1263-15»
    static String[] bookOf(Document doc) {
        for (String name : BOOK_FIELDS) {
            String v = doc.get(name);
            if (v != null && v.matches("\\d+")) return new String[] {v, "field «" + name + "»"};
        }
        for (IndexableField f : doc.getFields()) {
            Matcher m = KEY.matcher(value(f).trim());
            if (m.matches()) return new String[] {m.group(1), "field «" + f.name() + "» (book-page key)"};
        }
        return null;
    }

    static void copyBookFiles(Path books, Path out, Set<String> wanted) throws IOException {
        if (!Files.isDirectory(books)) return;
        try (var walk = Files.walk(books, 3)) {
            for (Path p : (Iterable<Path>) walk::iterator) {
                String name = p.getFileName().toString();
                if (name.endsWith(".db") && wanted.contains(name.substring(0, name.length() - 3)))
                    Files.copy(p, out.resolve(name), StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    static Writer open(Path p) {
        try {
            return new BufferedWriter(new OutputStreamWriter(Files.newOutputStream(p), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static String value(IndexableField f) {
        if (f.stringValue() != null) return f.stringValue();
        if (f.numericValue() != null) return f.numericValue().toString();
        if (f.binaryValue() != null) return f.binaryValue().utf8ToString();
        return "";
    }

    static String json(Document doc) {
        StringBuilder sb = new StringBuilder("{");
        Map<String, List<String>> fields = new LinkedHashMap<>();
        for (IndexableField f : doc.getFields()) fields.computeIfAbsent(f.name(), k -> new ArrayList<>()).add(value(f));
        boolean first = true;
        for (var e : fields.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            quote(sb, e.getKey()).append(':');
            if (e.getValue().size() == 1) quote(sb, e.getValue().get(0));
            else {
                sb.append('[');
                for (int i = 0; i < e.getValue().size(); i++) { if (i > 0) sb.append(','); quote(sb, e.getValue().get(i)); }
                sb.append(']');
            }
        }
        return sb.append('}').toString();
    }

    static StringBuilder quote(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> { if (c < 0x20) sb.append(String.format("\\u%04x", (int) c)); else sb.append(c); }
            }
        }
        return sb.append('"');
    }

    static String shorten(String s, int n) {
        s = s.replace('\n', ' ');
        return s.length() > n ? s.substring(0, n) + "…" : s;
    }
}
