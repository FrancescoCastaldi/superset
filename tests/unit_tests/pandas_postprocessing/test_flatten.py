# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pandas as pd

from superset.utils import pandas_postprocessing as pp
from superset.utils.pandas_postprocessing.utils import FLAT_COLUMN_SEPARATOR
from tests.unit_tests.fixtures.dataframes import timeseries_df


def test_flat_should_not_change():
    df = pd.DataFrame(
        data={
            "foo": [1, 2, 3],
            "bar": [4, 5, 6],
        }
    )

    assert pp.flatten(df).equals(df)


def test_flat_should_not_reset_index():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    df = pd.DataFrame(index=index, data={"foo": [1, 2, 3], "bar": [4, 5, 6]})

    assert pp.flatten(df, reset_index=False).equals(df)


def test_flat_should_flat_datetime_index():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    df = pd.DataFrame(index=index, data={"foo": [1, 2, 3], "bar": [4, 5, 6]})

    assert pp.flatten(df).equals(
        pd.DataFrame(
            {
                "__timestamp": index,
                "foo": [1, 2, 3],
                "bar": [4, 5, 6],
            }
        )
    )


def test_flat_should_flat_multiple_index():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    iterables = [["foo", "bar"], [1, "two"]]
    columns = pd.MultiIndex.from_product(iterables, names=["level1", "level2"])
    df = pd.DataFrame(index=index, columns=columns, data=1)

    assert pp.flatten(df).equals(
        pd.DataFrame(
            {
                "__timestamp": index,
                FLAT_COLUMN_SEPARATOR.join(["foo", "1"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["foo", "two"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["bar", "1"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["bar", "two"]): [1, 1, 1],
            }
        )
    )


def test_flat_should_drop_index_level():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    columns = pd.MultiIndex.from_arrays(
        [["a"] * 3, ["b"] * 3, ["c", "d", "e"], ["ff", "ii", "gg"]],
        names=["level1", "level2", "level3", "level4"],
    )
    df = pd.DataFrame(index=index, columns=columns, data=1)

    # drop level by index
    assert pp.flatten(
        df.copy(),
        drop_levels=(
            0,
            1,
        ),
    ).equals(
        pd.DataFrame(
            {
                "__timestamp": index,
                FLAT_COLUMN_SEPARATOR.join(["c", "ff"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["d", "ii"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["e", "gg"]): [1, 1, 1],
            }
        )
    )

    # drop level by name
    assert pp.flatten(df.copy(), drop_levels=("level1", "level2")).equals(
        pd.DataFrame(
            {
                "__timestamp": index,
                FLAT_COLUMN_SEPARATOR.join(["c", "ff"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["d", "ii"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["e", "gg"]): [1, 1, 1],
            }
        )
    )

    # only leave 1 level
    assert pp.flatten(df.copy(), drop_levels=(0, 1, 2)).equals(
        pd.DataFrame(
            {
                "__timestamp": index,
                FLAT_COLUMN_SEPARATOR.join(["ff"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["ii"]): [1, 1, 1],
                FLAT_COLUMN_SEPARATOR.join(["gg"]): [1, 1, 1],
            }
        )
    )


def test_flat_should_not_droplevel():
    assert pp.flatten(timeseries_df, drop_levels=(0,)).equals(
        pd.DataFrame(
            {
                "index": pd.to_datetime(
                    ["2019-01-01", "2019-01-02", "2019-01-05", "2019-01-07"]
                ),
                "label": ["x", "y", "z", "q"],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
    )


def test_flat_integer_column_name():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    columns = pd.MultiIndex.from_arrays(
        [["a"] * 3, [100, 200, 300]],
        names=["level1", "level2"],
    )
    df = pd.DataFrame(index=index, columns=columns, data=1)
    assert pp.flatten(df, drop_levels=(0,)).equals(
        pd.DataFrame(
            {
                "__timestamp": pd.to_datetime(
                    ["2021-01-01", "2021-01-02", "2021-01-03"]
                ),
                "100": [1, 1, 1],
                "200": [1, 1, 1],
                "300": [1, 1, 1],
            }
        )
    )


def test_escape_column_name():
    index = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    index.name = "__timestamp"
    columns = pd.MultiIndex.from_arrays(
        [
            ["level1,value1", "level1,value2", "level1,value3"],
            ["level2, value1", "level2, value2", "level2, value3"],
        ],
        names=["level1", "level2"],
    )
    df = pd.DataFrame(index=index, columns=columns, data=1)
    assert list(pp.flatten(df).columns.values) == [
        "__timestamp",
        "level1\\,value1" + FLAT_COLUMN_SEPARATOR + "level2\\, value1",
        "level1\\,value2" + FLAT_COLUMN_SEPARATOR + "level2\\, value2",
        "level1\\,value3" + FLAT_COLUMN_SEPARATOR + "level2\\, value3",
    ]


def test_flat_should_reset_multilevel_row_index():
    index = pd.MultiIndex.from_tuples(
        [("group_a", "sub_1"), ("group_a", "sub_2"), ("group_b", "sub_1")],
        names=["group", "subgroup"],
    )
    df = pd.DataFrame(index=index, data={"val": [10, 20, 30]})
    res = pp.flatten(df)
    assert list(res.columns.values) == ["group", "subgroup", "val"]
    assert res["group"].tolist() == ["group_a", "group_a", "group_b"]
    assert res["subgroup"].tolist() == ["sub_1", "sub_2", "sub_1"]
    assert res["val"].tolist() == [10, 20, 30]


def test_flat_should_flat_row_and_column_multiindex() -> None:
    index = pd.MultiIndex.from_tuples(
        [("US", "web"), ("DE", "mobile")], names=["country", "channel"]
    )
    columns = pd.MultiIndex.from_tuples([("sales", "sum")])
    df = pd.DataFrame([10, 20], index=index, columns=columns)
    metric = FLAT_COLUMN_SEPARATOR.join(["sales", "sum"])
    expected = pd.DataFrame(
        {"country": ["US", "DE"], "channel": ["web", "mobile"], metric: [10, 20]}
    )

    result = pp.flatten(df)

    pd.testing.assert_frame_equal(result, expected)
    assert result.to_dict(orient="records") == expected.to_dict(orient="records")


def test_flat_should_preserve_row_multiindex_when_reset_disabled() -> None:
    index = pd.MultiIndex.from_tuples(
        [("US", "web"), ("DE", "mobile")], names=["country", "channel"]
    )
    df = pd.DataFrame({"sales": [10, 20]}, index=index)

    pd.testing.assert_frame_equal(pp.flatten(df, reset_index=False), df)
