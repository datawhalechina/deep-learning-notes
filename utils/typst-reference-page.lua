function Pandoc(doc)
  if FORMAT:match("^typst") then
    doc.blocks:insert(pandoc.RawBlock("typst", "#pagebreak(weak: true)"))
  end

  return doc
end
