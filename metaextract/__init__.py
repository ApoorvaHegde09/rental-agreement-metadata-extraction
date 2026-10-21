"""metaextract — learned (non rule-based) metadata extraction from rental agreements."""
FIELDS = ["value", "start", "end", "notice", "party_one", "party_two"]
CSV_COLUMNS = {
    "value": "Aggrement Value",
    "start": "Aggrement Start Date",
    "end": "Aggrement End Date",
    "notice": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}
