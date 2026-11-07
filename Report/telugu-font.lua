-- Wrap Telugu code points so wkhtmltopdf can use Noto Sans Telugu.
-- Qt WebKit does not fall back per character inside one font-family list.

local function is_telugu(cp)
  return cp >= 0x0C00 and cp <= 0x0C7F
end

local function has_telugu(text)
  for _, cp in utf8.codes(text) do
    if is_telugu(cp) then
      return true
    end
  end
  return false
end

local function html_escape(text)
  text = text:gsub("&", "&amp;")
  text = text:gsub("<", "&lt;")
  text = text:gsub(">", "&gt;")
  return text
end

local function mixed_html(text)
  local parts = {}
  local buf = {}
  local mode = nil
  local function flush()
    if #buf == 0 then
      return
    end
    local chunk = html_escape(table.concat(buf))
    if mode == "te" then
      table.insert(parts, '<span class="te">' .. chunk .. "</span>")
    else
      table.insert(parts, chunk)
    end
    buf = {}
  end
  for _, cp in utf8.codes(text) do
    local next_mode = is_telugu(cp) and "te" or "other"
    if mode ~= nil and next_mode ~= mode then
      flush()
    end
    mode = next_mode
    table.insert(buf, utf8.char(cp))
  end
  flush()
  return table.concat(parts)
end

function Str(el)
  if not has_telugu(el.text) then
    return nil
  end
  return pandoc.RawInline("html", mixed_html(el.text))
end

function Code(el)
  if not has_telugu(el.text) then
    return nil
  end
  return pandoc.RawInline("html", "<code>" .. mixed_html(el.text) .. "</code>")
end

function CodeBlock(el)
  if not has_telugu(el.text) then
    return nil
  end
  return pandoc.RawBlock("html", "<pre><code>" .. mixed_html(el.text) .. "</code></pre>")
end
