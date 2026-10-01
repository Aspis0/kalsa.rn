#!/usr/bin/env python3
"""Print the string a defined symbol binds to in an ELF32 little-endian file.

The QDSP6 HTP skels are ELF32 LE; scripts/assert-htp-skels.sh reads their
identity stamps through this resolver instead of grepping raw bytes, so an
appended look-alike string cannot vouch for anything -- only the bytes the
symbol actually points at count.

nm/objdump are not portable here: macOS ships neither in an ELF-capable form
and the slim ubuntu runner image has no binutils, so this parses the section
headers and .symtab directly. It is enough to resolve one symbol: file offset
= sh_offset + (st_value - sh_addr) of the section st_shndx names.

Fails (exit 1, message on stderr) when the file is not ELF32 LE, has no
.symtab, or the symbol has zero or multiple definitions.
"""
import struct
import sys

SHT_SYMTAB = 2
SHN_UNDEF = 0
SHN_LORESERVE = 0xFF00

ELF32_EHDR = 52
ELF32_SHDR = 40
ELF32_SYM = 16


def fail(msg):
    print(f"htp_elf_symbol: {msg}", file=sys.stderr)
    sys.exit(1)


def cstr(data, off, end, what):
    nul = data.find(b"\0", off, end)
    if nul < 0:
        fail(f"{what} is not NUL-terminated where the section ends")
    return data[off:nul].decode("utf-8", "replace")


def main():
    if len(sys.argv) != 3:
        fail("usage: htp_elf_symbol.py <elf32-file> <symbol>")
    path, want = sys.argv[1], sys.argv[2]
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        fail(f"cannot read {path}: {e}")

    if len(data) < ELF32_EHDR or data[:4] != b"\x7fELF":
        fail(f"{path}: not an ELF file")
    if data[4] != 1 or data[5] != 1:
        fail(f"{path}: not a 32-bit little-endian ELF (QDSP6 skels are)")
    e_shoff = struct.unpack_from("<I", data, 0x20)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x2E)[0]
    e_shnum = struct.unpack_from("<H", data, 0x30)[0]
    if e_shoff == 0 or e_shnum == 0:
        fail(f"{path}: no section headers")

    sections = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        # name, type, flags, addr, offset, size, link, info, align, entsize
        sections.append(struct.unpack_from("<IIIIIIIIII", data, base))

    symtabs = [s for s in sections if s[1] == SHT_SYMTAB]
    if not symtabs:
        fail(f"{path}: no .symtab section")

    defs = []
    for symtab in symtabs:
        strtab = sections[symtab[6]]  # sh_link names the string table
        entsize = symtab[9] or ELF32_SYM
        for i in range(symtab[5] // entsize):
            base = symtab[4] + i * entsize
            st_name, st_value, _size, _info, _other, st_shndx = struct.unpack_from(
                "<IIIBBH", data, base)
            if st_shndx == SHN_UNDEF or st_shndx >= SHN_LORESERVE:
                continue
            str_off, str_size = strtab[4], strtab[5]
            name = cstr(data, str_off + st_name, str_off + str_size,
                        f"symbol name {st_name} in .strtab")
            if name == want:
                defs.append((st_value, st_shndx))

    if not defs:
        fail(f"{path}: {want} has 0 definitions")
    if len(defs) > 1:
        fail(f"{path}: {want} has {len(defs)} definitions -- refusing to guess")

    st_value, shndx = defs[0]
    if shndx >= len(sections):
        fail(f"{path}: {want} names section {shndx}, past the header table")
    sec = sections[shndx]
    file_off = sec[4] + (st_value - sec[3])
    if not sec[4] <= file_off < sec[4] + sec[5]:
        fail(f"{path}: {want} address 0x{st_value:x} is outside its section")
    print(cstr(data, file_off, sec[4] + sec[5], f"{want} bytes").strip())


if __name__ == "__main__":
    main()
