var ablat_video_items = [
    { video: "assets/morphing/video_results/ablat1.mp4", caption: "" },
    { video: "assets/morphing/video_results/ablat2.mp4", caption: "" },
    { video: "assets/morphing/video_results/ablat3.mp4", caption: "" }
];

var app1_video_items = [
    { video: "assets/morphing/video_results/app1_1.mp4", caption: "" },
    { video: "assets/morphing/video_results/app1_2.mp4", caption: "" }
];

var app2_video_items = [
    { video: "assets/morphing/video_results/app2_1.mp4", caption: "" },
    { video: "assets/morphing/video_results/app2_2.mp4", caption: "" }
];

var app3_video_items = [
    { video: "assets/morphing/video_results/app3.mp4", caption: "" }
];

var gener_video_items = [
    { video: "assets/morphing/video_results/gener1.mp4", caption: "" },
    { video: "assets/morphing/video_results/gener2.mp4", caption: "" }
];

var comp_video_items = [
    { video: "assets/morphing/video_results/comp1.mp4", caption: "" },
    { video: "assets/morphing/video_results/comp2.mp4", caption: "" },
    { video: "assets/morphing/video_results/comp3.mp4", caption: "" },
    { video: "assets/morphing/video_results/comp4.mp4", caption: "" }
];

var ip_video_items = [
    { video: "assets/morphing/video_results/ip1.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip2.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip3.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip4.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip5.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip6.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip7.mp4", caption: "" },
    { video: "assets/morphing/video_results/ip8.mp4", caption: "" }
];

var obj_video_items = [
    { video: "assets/morphing/video_results/obj1.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj2.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj3.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj4.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj5.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj6.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj7.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj8.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj9.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj10.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj11.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj12.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj13.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj14.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj15.mp4", caption: "" },
    { video: "assets/morphing/video_results/obj16.mp4", caption: "" }

];

function video_carousel_item_template(item) {
    return `<div style="width: 100%; display: flex; flex-direction: column; align-items: center; background: white; border-radius: 8px;">
                <video controls playsinline loop muted autoplay preload="metadata"
                    onloadeddata="this.style.opacity = 1"
                    style="width: 100%; height: auto; display: block; object-fit: contain; opacity: 0; transition: opacity 0.25s ease; border-radius: 8px;"
                    src="${item.video}"></video>
            </div>`;
}
