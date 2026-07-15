from pathlib import Path

p = Path("app.py")
s = p.read_text(encoding="utf-8")

old = "    st.dataframe(display_df[cols] if cols else display_df, use_container_width=True, height=620)"

new = """    display_df = display_df.loc[:, ~display_df.columns.duplicated()].copy()
    cols = [c for c in cols if c in display_df.columns]
    cols = list(dict.fromkeys(cols))
    view_df = display_df[cols] if cols else display_df
    st.dataframe(view_df, use_container_width=True, height=620)"""

if old not in s:
    raise SystemExit("Could not find the target st.dataframe line. Open app.py and patch manually around the st.dataframe(display_df[cols]...) line.")

p.write_text(s.replace(old, new), encoding="utf-8")
print("Patched app.py: duplicate display columns will be removed.")
