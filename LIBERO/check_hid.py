import hid

for d in hid.enumerate():
    if d.get("vendor_id") == 0x256F:
        print(
            hex(d["vendor_id"]),
            hex(d["product_id"]),
            d.get("product_string"),
            "usage_page=", hex(d.get("usage_page", 0)),
            "usage=", hex(d.get("usage", 0)),
        )

