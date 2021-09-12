import json

def np_save_as_json(data, filename):
    assert type(data) == dict or type(data) == list

    if type(data) == dict:
        for k, val in data.items():
            if type(val) == list:
                for i, obj in enumerate(val):
                    val[i] = obj.tolist()
                data[k] = val

    with open(filename, "w") as f:
        json.dump(data, f)