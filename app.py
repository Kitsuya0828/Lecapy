import datetime
import tempfile
import uuid
import os
from pathlib import Path
from collections import deque

import cv2
import img2pdf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from pathlib import Path

main_image = Image.open('main.jpg')
st.image(main_image)

"""
## Let's get started!

誰でも簡単に**授業動画をPDFに変換**してダウンロード。\n
**Deep Learning**で重複したフレームを自動的に削除。\n
試験対策や書き込み用に**最適なPDF**が、すぐにツクレル。\n

1. 授業動画をスマホまたはPCにダウンロードする
2. 動画のサイズが200MB以下であることを確認する(※1)
3. 動画をアップロードして自動的に完成したPDFをダウンロードする
"""
uploaded_file = st.file_uploader("※アップロードする動画サイズの上限は200MBです", type=["mp4", "mov", "m4a"])
"""
---
"""


def capture_frame(video_path: str, step_frame: int, UUID: str):
    cap = cv2.VideoCapture(video_path)
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        return
    
    digit = len(str(video_frame_count)) # 総フレーム数の桁数

    # プログレスバー
    text_cap = st.empty()
    bar_cap = st.progress(0)

    step_num = len(range(0, video_frame_count, step_frame))
    index = 0

    for n in range(0, video_frame_count, step_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()

        index += 1
        text_cap.text(f'キャプチャを抽出中 {int(100*index/step_num)}%')
        bar_cap.progress(index/step_num)

        if ret:
            cv2.imwrite(f'static/{UUID}/{str(n).zfill(digit)}.jpg', frame)


def deduplicate_frame(UUID: str, threshold: int, frame_size: tuple, scale: int):
    IMG_DIR = f'static/{UUID}/'
    IMG_SIZE = tuple(map(lambda x: x//scale, frame_size))

    dump_list = []
    files = deque(os.listdir(Path(IMG_DIR).absolute()))    # 逆順から類似判定
    num_files = len(files)

    index = 1
    # プログレスバー
    text_judge = st.empty()
    bar_judge = st.progress(index/num_files)

    while files:
        target_file = files.popleft()
        target_img_path = Path(IMG_DIR + target_file).absolute()
        target_img = cv2.imread(str(target_img_path), cv2.IMREAD_GRAYSCALE)
        if scale != 1:
            target_img = cv2.resize(target_img, IMG_SIZE)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)

        same_slide = True
        while same_slide:
            if not files:
                break

            comparing_file = files.popleft()
            comparing_img_path = Path(IMG_DIR + comparing_file).absolute()

            try:
                comparing_img = cv2.imread(str(comparing_img_path), cv2.IMREAD_GRAYSCALE)
                if scale != 1:
                    comparing_img = cv2.resize(comparing_img, IMG_SIZE)

                (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
                matches = bf.match(target_des, comparing_des)

                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)
            except cv2.error:
                ret = 10**6

            index += 1
            bar_judge.progress(index/num_files)
            text_judge.text(f'類似度を解析中 {int(100*index/num_files)}%')

            if ret < threshold:
                dump_list.append(comparing_file)
            else:
                files.appendleft(comparing_file)    # 次のターゲット画像
                same_slide = False
    return dump_list[::-1]


def generate_pdf(UUID: str, dump_list: list):
    img_folder = f'static/{UUID}/'
    img_folder_path = Path(img_folder).absolute()
    save_path = Path(f'static/{UUID}.pdf').absolute()
    with open(str(save_path), "wb") as f:
        pdf_file_path_list = []
        for file in os.listdir(img_folder_path):
            if file not in dump_list:
                path = str(Path(img_folder+file).absolute())
                pdf_file_path_list.append(path)
        f.write(img2pdf.convert(pdf_file_path_list))


if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # サイドバー
    st.sidebar.markdown("# カスタマイズオプション")

    option = st.sidebar.selectbox('フレームのリサイズ倍率（リサイズすると若干精度が落ちますが、処理が高速化します）', ('1倍(リサイズしない)', '2倍', '4倍'))
    scale = int(option[0])

    threshold = st.sidebar.slider('類似度判定基準(数値が大きいほど「似ている」のハードルが低い)', min_value=0, max_value=100, value=30)

    interval_frame_count = st.sidebar.slider("キャプチャ間隔(フレーム)", min_value=10, max_value=video_frame_count//10, value=300)
    td = datetime.timedelta(seconds=interval_frame_count / video_fps)
    m, s = divmod(td.seconds, 60)
    interval_range = f'{m}分{s}秒' if m else f'{s}秒'
    st.sidebar.write('↪', interval_frame_count, 'フレーム', '÷', f'{video_fps} FPS', '=', f'{interval_range} 間隔')

    UUID = uuid.uuid4()  # ユニークなID
    
    path_2_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    path_2_uuid = os.path.join(path_2_static, str(UUID))

    os.mkdir(path_2_uuid)
    capture_frame(tfile.name, interval_frame_count, UUID)
    dump_list = deduplicate_frame(UUID, threshold, (video_height, video_width), scale)
     
    root = f"static/{UUID}"
    lsdir = os.listdir(Path(root).absolute())
    imgs = []
    text_plot = st.empty()
    bar_plot = st.progress(0)
    step_num = len(lsdir)
    index = 1
    for l in lsdir:
        target = os.path.join(root, l)
        height, width = video_height, video_width

        if l in dump_list:  # 似ているフレーム
            img = cv2.imread(str(Path(target).absolute()), cv2.IMREAD_GRAYSCALE)
            img = cv2.bitwise_not(img)
        else:
            img = cv2.imread(str(Path(target).absolute()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.putText(img, str(index), (0, height//2), cv2.FONT_HERSHEY_PLAIN, width*height/(200**2), (255, 0, 0), 20, cv2.LINE_AA)
            index += 1
        imgs.append(img)

    shownumber = len(lsdir)
    showaxis = 1

    while showaxis*showaxis < shownumber:
        showaxis += 1

    cnt = 0
    while True:
        if cnt >= shownumber:
            break

        fig, axs = plt.subplots(showaxis, showaxis)
        ar = axs.ravel()
        for i in range(showaxis*showaxis):
            ar[i].axis('off')
            if i < shownumber:
                ar[i].imshow(imgs[cnt])
                cnt += 1

                bar_plot.progress(cnt/step_num)
                text_plot.text(f'イメージをプロット中 {int(100*cnt/step_num)}%')
    st.pyplot(fig)

    generate_pdf(UUID, dump_list)
    st.markdown("※ 反転していない画像がダウンロードPDFに含まれます")

    with open(str(Path(f"static/{UUID}.pdf").absolute()), "rb") as file:
        btn = st.download_button(
            label="Download PDF",
            data=file,
            file_name="download.pdf"
        )

    st.markdown("## Are you satisfied?")
    st.markdown("各ページの取捨選択は、PDFダウンロード後に以下のサイトをご利用ください")
    st.markdown('[オンラインでPDFからページを削除](https://www.ilovepdf.com/ja/remove-pages)')

else:
    """
    ---

    ※1 : 動画のサイズが大きすぎる場合は、スマホやPCで分割したり画面サイズを変更したりするのがオススメです

    <便利なサイト> [動画のファイルサイズをオンラインで小さくする、動画をオンラインで圧縮する \(MP4, AVI, MOV, MPEG\) \| VideoSmaller](https://www.videosmaller.com/jp/)
    """
st.markdown("")
"""
---
"""
st.markdown("© 2022 [Adumaru Channel](https://www.youtube.com/channel/UC00vvtUdtiche9vz_S4UjhQ)")
