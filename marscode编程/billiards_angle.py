import math

def main():
    print("""
            ****************************模拟计算击打台球的出杆角度****************************
            角度值为 击打球与碰撞球所在直线 与 碰撞球与网洞所在直线  形成的 锐角 单位为 °
            distance距离值为击打球与碰撞球之间的距离 单位按球的半径为 1 计算
            distance距离越小目测值误差影响越大 a角度越小目测之误差影响越大
    """)

    while True:
        a = float(input("角度（锐角）: "))
        # 计算前提
        if a >= 90:
            print("\n输入有误 不可击中")
        else:
            distance = float(input("距离（球半径为1）: "))
            a = math.radians(a)
            pi = math.radians(180)
            fcd = math.atan((0.5 * distance - math.cos(a)) / math.sin(a))
            # 满足条件
            if (a < 0.5 * pi) and fcd > a:
                # 计算过程
                relta = 2 * math.sin(a + math.atan(2 * math.sin(a) / (distance - 2 * math.cos(a))))
                reltdis = 90 - fcd / pi * 180

                # 取两位小数
                resulta = round(relta, 2)
                resultdis = round(reltdis, 2)

                print(f"角度偏移 {resultdis}°")
                print(f"位置偏移 {resulta}")
            else:
                print("\n输入有误 不可击中")
        
        continue_choice = int(input("\n继续1 停止0: "))
        if continue_choice != 1:
            break

    print("\n程序结束")

if __name__ == "__main__":
    main()
