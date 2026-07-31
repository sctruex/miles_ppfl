def make_f1_lsts(x):
    x = x.replace("array(", "")
    x = x.replace(",dtype=float32)", "")
    for char in " \t\n":
        x = x.replace(char, "")
    print(x)
    x = x.strip("[]")
    str_lsts = x.split("]")
    f1_lsts = []
    for f1_str in str_lsts:
        f1_vals = []
        vals = f1_str.strip(",[").split(",")
        for val in vals:
            if val != "":
                f1_vals.append(float(val))
        f1_lsts.append(f1_vals)
    return f1_lsts


def main():
    x = "[array([0.4945478 , 0.4826468 , 0.        , 0.00541028, 0.        ,0.4838068 , 0.31316984, 0.41248685, 0.21661238, 0.3811466 ],dtype=float32), array([0.0180624 , 0.33212247, 0.        , 0.01152738, 0.04815133,0.46984795, 0.31327605, 0.6319116 , 0.40832815, 0.5891938 ],dtype=float32)]"
    f1_lst = make_f1_lsts(x)
    print(f1_lst, type(f1_lst), type(f1_lst[0]), type(f1_lst[0][0]))


if __name__ == "__main__":
    main()
