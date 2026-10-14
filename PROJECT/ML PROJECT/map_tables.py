import docx

doc = docx.Document(r'C:\Users\Madhav\Downloads\AML_GNN_Research_Paper.docx')

body_elements = []
for child in doc.element.body:
    tag = child.tag.split('}')[-1]
    if tag == 'p':
        # get text from paragraph element
        p = docx.text.paragraph.Paragraph(child, doc)
        if p.text.strip():
            body_elements.append(('p', p.text.strip()))
    elif tag == 'tbl':
        body_elements.append(('tbl', ''))

table_counter = 0
with open('tables_with_captions.txt', 'w', encoding='utf-8') as out:
    for idx, (kind, val) in enumerate(body_elements):
        if kind == 'tbl':
            caption = 'NO CAPTION'
            for j in range(idx-1, max(-1, idx-6), -1):
                if body_elements[j][0] == 'p' and body_elements[j][1]:
                    caption = body_elements[j][1]
                    break
            t = doc.tables[table_counter]
            out.write(f"=== TABLE {table_counter} (Caption: {caption}) ===\n")
            for r_i, r in enumerate(t.rows):
                row_vals = [c.text.strip().replace('\n', ' ') for c in r.cells]
                out.write(f"  Row {r_i}: {row_vals}\n")
            out.write("\n")
            table_counter += 1

print(f"Done mapping {table_counter} tables.")
