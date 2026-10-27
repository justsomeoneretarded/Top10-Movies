from movie_pipeline.availability import availability_line


def test_aggregator_channels_are_not_named():
    line = availability_line({"status": "found", "subscription": ["Amazon Prime Video"], "free": ["JustWatch TV"]})
    assert "JustWatch" not in line and "Prime Video" in line
    assert availability_line({"status": "found", "free": ["JustWatch TV"]}) == ""
