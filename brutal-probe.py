import argparse
import json
import os
import random
import socket
import struct
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

TCP_BRUTAL_PARAMS = 23301
TCP_CONGESTION = 13


class TestPhase(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"


class TcpBrutalSocket:
    def __init__(self, sock):
        sock.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, "brutal".encode())
        self.sock = sock
        self.gain = 1.5
        self.rate = 62500  # byte/s
        # PS: rate是本端发送到对端的速度，比如你在服务器上设置这个就是链接的下载速度
        self.flush_config()

    def flush_config(self):
        value = struct.pack("qi", int(self.rate), int(self.gain * 10))
        self.sock.setsockopt(socket.IPPROTO_TCP, TCP_BRUTAL_PARAMS, value)

    def set_rate(self, rate: float):
        """设置发送速率（字节/秒）"""
        self.rate = rate
        self.flush_config()

    def sendall(self, bytes):
        self.sock.sendall(bytes)

    def recvall(self, num):
        data = b""
        while len(data) < num:
            expected = num - len(data)
            chunk = self.sock.recv(expected)
            if not chunk:
                raise ConnectionError("Unexpected EOF")
            data += chunk
        return data

    def send_json(self, obj):
        s = json.dumps(obj)
        bytes = s.encode("utf-8")
        data = struct.pack(">i", len(bytes))
        self.sendall(data)
        self.sendall(bytes)

    def recv_json(self):
        length = struct.unpack(">i", self.recvall(4))[0]
        data = self.recvall(length)
        s = data.decode("utf-8")
        return json.loads(s)

    def send_time(self):
        stamp = time.time()
        self.sendall(struct.pack(">q", int(stamp * 1000)))

    def recv_time(self):
        return float(struct.unpack(">q", self.recvall(8))[0]) / 1000

    def send_data(self, size: int) -> float:
        """发送指定大小的随机数据，返回传输时间"""
        start_time = time.time()

        # 分块发送以避免内存占用过大
        chunk_size = min(1024 * 1024, size)  # 1MB chunks
        sent = 0

        # 生成随机数据避免压缩
        random_chunk = os.urandom(chunk_size)

        while sent < size:
            to_send = min(chunk_size, size - sent)
            if to_send == chunk_size:
                self.sendall(random_chunk)
            else:
                self.sendall(os.urandom(to_send))
            sent += to_send

        elapsed = time.time() - start_time
        return elapsed

    def recv_data(self, size: int) -> float:
        """接收指定大小的数据，返回传输时间"""
        start_time = time.time()

        # 分块接收
        chunk_size = min(1024 * 1024, size)  # 1MB chunks
        received = 0

        while received < size:
            to_recv = min(chunk_size, size - received)
            data = self.recvall(to_recv)
            received += to_recv

        elapsed = time.time() - start_time
        return elapsed

    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()


class BandwidthProbe:
    """带宽探测器"""

    def __init__(self, sock: TcpBrutalSocket, test_duration: float = 15.0):
        self.sock = sock
        self.test_duration = test_duration  # 每次测试的目标时长（秒）
        self.min_rate = 62500  # 最小速率 62 KB/s(内核实现限制)
        self.max_rate = 10 * 1024 * 1024 * 1024  # 最大速率 10 GB/s

        # 探测结果
        self.upload_rate = 62500  # 初始上传速率估计 62KB/s
        self.download_rate = 62500  # 初始下载速率估计 62KB/s

        # 探测历史
        self.upload_history = []
        self.download_history = []

        # 当前探测范围
        self.upload_min = self.min_rate
        self.upload_max = self.max_rate
        self.download_min = self.min_rate
        self.download_max = self.max_rate

        # 收敛判定
        self.convergence_threshold = 0.05  # 5% 的误差范围
        self.max_iterations = 256  # 最大迭代次数

        # 探测阶段标记
        self.upload_phase = "exploration"  # "exploration" or "convergence"
        self.download_phase = "exploration"  # "exploration" or "convergence"

    def format_speed(self, bytes_per_sec: float) -> str:
        """格式化速度显示"""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.1f} B/s"
        elif bytes_per_sec < 1024 * 1024:
            return f"{bytes_per_sec / 1024:.1f} KB/s"
        elif bytes_per_sec < 1024 * 1024 * 1024:
            return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
        else:
            return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"

    def test_upload(self, rate_estimate: float) -> Tuple[float, float]:
        """测试上传速度
        返回: (实际速率, 实际时间/预期时间比率)
        """
        data_size = int(rate_estimate * self.test_duration)

        print(
            f"  Testing upload at {self.format_speed(rate_estimate)}, data size: {data_size / 1024:.1f} KB"
        )

        # 发送测试命令
        command = {"action": "upload", "size": data_size, "rate": rate_estimate}
        self.sock.send_json(command)

        # 设置本地发送速率
        self.sock.set_rate(rate_estimate)

        # 发送数据并计时
        actual_time = self.sock.send_data(data_size)

        # 接收服务器响应
        response = self.sock.recv_json()

        if not response.get("success"):
            raise Exception(
                f"Upload test failed: {response.get('error', 'Unknown error')}"
            )

        server_time = response["actual_time"]
        actual_rate = data_size / server_time
        time_ratio = server_time / self.test_duration

        print(
            f"  Result: {self.format_speed(actual_rate)}, time ratio: {time_ratio:.2f}"
        )

        return actual_rate, time_ratio

    def test_download(self, rate_estimate: float) -> Tuple[float, float]:
        """测试下载速度
        返回: (实际速率, 实际时间/预期时间比率)
        """
        data_size = int(rate_estimate * self.test_duration)

        print(
            f"  Testing download at {self.format_speed(rate_estimate)}, data size: {data_size / 1024:.1f} KB"
        )

        # 发送测试命令
        command = {"action": "download", "size": data_size, "rate": rate_estimate}
        self.sock.send_json(command)

        # 接收数据并计时
        actual_time = self.sock.recv_data(data_size)

        # 接收服务器响应
        response = self.sock.recv_json()

        if not response.get("success"):
            raise Exception(
                f"Download test failed: {response.get('error', 'Unknown error')}"
            )

        actual_rate = data_size / actual_time
        time_ratio = actual_time / self.test_duration

        print(
            f"  Result: {self.format_speed(actual_rate)}, time ratio: {time_ratio:.2f}"
        )

        return actual_rate, time_ratio

    def adjust_estimate(
        self,
        current_estimate: float,
        actual_rate: float,
        time_ratio: float,
        min_bound: float,
        max_bound: float,
        phase: str,
    ) -> Tuple[float, float, float, str]:
        """根据测试结果调整估计值
        返回: (新估计值, 新最小边界, 新最大边界, 新阶段)
        """

        if phase == "exploration":
            # 探索阶段：快速增长找上限
            if time_ratio > 1.5:
                # 找到上限了，切换到收敛阶段
                new_max = current_estimate
                new_min = min_bound
                new_estimate = (new_min + new_max) / 2
                print(f"  Found upper limit! Switching to convergence phase.")
                print(
                    f"  Range: [{self.format_speed(new_min)}, {self.format_speed(new_max)}]"
                )
                return new_estimate, new_min, new_max, "convergence"
            else:
                # 还没到上限，继续探索，使用1.5倍增长
                new_estimate = current_estimate * 1.5
                new_min = current_estimate
                new_max = max_bound
                print(
                    f"  Exploration phase: increasing by 1.5x to {self.format_speed(new_estimate)}"
                )
                return new_estimate, new_min, new_max, "exploration"

        else:  # convergence phase
            # 收敛阶段：二分搜索精确定位
            # 如果时间比率在合理范围内（0.8-1.2），认为估计准确
            if 0.8 <= time_ratio <= 1.2:
                # 收敛，使用实际速率
                return actual_rate, actual_rate * 0.9, actual_rate * 1.1, "convergence"

            # 如果实际时间远大于预期（>1.5），说明估计过大
            elif time_ratio > 1.5:
                # 估计过大，降低上界
                new_max = current_estimate
                new_min = min_bound
                new_estimate = (new_min + new_max) / 2
                print(
                    f"  Estimate too high, adjusting range: [{self.format_speed(new_min)}, {self.format_speed(new_max)}]"
                )
                return new_estimate, new_min, new_max, "convergence"

            # 如果实际时间接近或小于预期（<=1.5），可能估计过小或接近实际值
            else:
                # 由于拥塞控制的影响，当估计过小时，实际时间不会显著小于预期
                # 所以我们需要继续提高估计值来探索
                if time_ratio <= 1.2:
                    # 可能估计过小，提高下界
                    new_min = current_estimate
                    new_max = max_bound
                    # 使用二分搜索
                    new_estimate = (new_min + new_max) / 2
                    print(
                        f"  Estimate might be too low, adjusting range: [{self.format_speed(new_min)}, {self.format_speed(new_max)}]"
                    )
                else:
                    # 接近实际值，微调
                    new_min = current_estimate * 0.8
                    new_max = current_estimate * 1.2
                    new_estimate = actual_rate
                    print(
                        f"  Close to actual, fine-tuning: [{self.format_speed(new_min)}, {self.format_speed(new_max)}]"
                    )

                return new_estimate, new_min, new_max, "convergence"

    def is_converged(self, min_bound: float, max_bound: float) -> bool:
        """判断是否已收敛"""
        if max_bound <= min_bound:
            return True

        ratio = (max_bound - min_bound) / max_bound
        return ratio <= self.convergence_threshold

    def run_probe(self):
        """运行完整的带宽探测"""
        print("\n" + "=" * 50)
        print("Starting bandwidth probe...")
        print(
            "Two-phase algorithm: Exploration (1.5x growth) → Convergence (binary search)"
        )
        print("=" * 50)

        iteration = 0
        upload_converged = False
        download_converged = False

        while iteration < self.max_iterations and not (
            upload_converged and download_converged
        ):
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # 上传测试
            if not upload_converged:
                print(f"\nUpload test:")
                try:
                    actual_rate, time_ratio = self.test_upload(self.upload_rate)
                    self.upload_history.append({
                        "estimate": self.upload_rate,
                        "actual": actual_rate,
                        "time_ratio": time_ratio,
                    })

                    # 调整估计值
                    (
                        self.upload_rate,
                        self.upload_min,
                        self.upload_max,
                        self.upload_phase,
                    ) = self.adjust_estimate(
                        self.upload_rate,
                        actual_rate,
                        time_ratio,
                        self.upload_min,
                        self.upload_max,
                        self.upload_phase,
                    )

                    # 检查是否收敛（只在收敛阶段检查）
                    if self.upload_phase == "convergence" and self.is_converged(
                        self.upload_min, self.upload_max
                    ):
                        upload_converged = True
                        print(
                            f"  Upload speed converged: {self.format_speed(self.upload_rate)}"
                        )

                except Exception as e:
                    print(f"  Upload test failed: {e}")

            # 下载测试
            if not download_converged:
                print(f"\nDownload test:")
                try:
                    actual_rate, time_ratio = self.test_download(self.download_rate)
                    self.download_history.append({
                        "estimate": self.download_rate,
                        "actual": actual_rate,
                        "time_ratio": time_ratio,
                    })

                    # 调整估计值
                    (
                        self.download_rate,
                        self.download_min,
                        self.download_max,
                        self.download_phase,
                    ) = self.adjust_estimate(
                        self.download_rate,
                        actual_rate,
                        time_ratio,
                        self.download_min,
                        self.download_max,
                        self.download_phase,
                    )

                    # 检查是否收敛（只在收敛阶段检查）
                    if self.download_phase == "convergence" and self.is_converged(
                        self.download_min, self.download_max
                    ):
                        download_converged = True
                        print(
                            f"  Download speed converged: {self.format_speed(self.download_rate)}"
                        )

                except Exception as e:
                    print(f"  Download test failed: {e}")

            # 延迟避免过于频繁
            time.sleep(1)

        # 输出最终结果
        self.print_results()

    def print_results(self):
        """打印探测结果"""
        print("\n" + "=" * 50)
        print("Bandwidth Probe Results")
        print("=" * 50)

        print(f"\nFinal Upload Speed:   {self.format_speed(self.upload_rate)}")
        print(f"Final Download Speed: {self.format_speed(self.download_rate)}")

        if self.upload_history:
            print(f"\nUpload test iterations: {len(self.upload_history)}")
            for i, record in enumerate(self.upload_history[-3:], 1):  # 显示最后3次
                print(
                    f"  {i}. Estimate: {self.format_speed(record['estimate'])}, "
                    f"Actual: {self.format_speed(record['actual'])}, "
                    f"Time ratio: {record['time_ratio']:.2f}"
                )

        if self.download_history:
            print(f"\nDownload test iterations: {len(self.download_history)}")
            for i, record in enumerate(self.download_history[-3:], 1):  # 显示最后3次
                print(
                    f"  {i}. Estimate: {self.format_speed(record['estimate'])}, "
                    f"Actual: {self.format_speed(record['actual'])}, "
                    f"Time ratio: {record['time_ratio']:.2f}"
                )

        print("\n" + "=" * 50)

    def generate_plots(self) -> str:
        """生成散点图并保存为HTML文件"""
        # 创建子图：1行2列
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Upload Speed Test Results", "Download Speed Test Results"),
            horizontal_spacing=0.15,
        )

        # 准备数据
        if self.upload_history:
            upload_estimates = [record["estimate"] for record in self.upload_history]
            upload_actuals = [record["actual"] for record in self.upload_history]
            upload_time_ratios = [
                record["time_ratio"] for record in self.upload_history
            ]

            # 创建hover text
            upload_hover = [
                f"Estimate: {self.format_speed(est)}<br>"
                + f"Actual: {self.format_speed(act)}<br>"
                + f"Time Ratio: {tr:.2f}<br>"
                + f"Test #{i + 1}"
                for i, (est, act, tr) in enumerate(
                    zip(upload_estimates, upload_actuals, upload_time_ratios)
                )
            ]

            # 添加上传散点图
            fig.add_trace(
                go.Scatter(
                    x=upload_estimates,
                    y=upload_actuals,
                    mode="markers",
                    name="Upload Tests",
                    marker=dict(
                        size=10,
                        color=upload_time_ratios,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Time Ratio", x=0.45, len=0.5),
                    ),
                    text=upload_hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # 添加理想线 (y=x)
            max_upload = max(max(upload_estimates), max(upload_actuals))
            fig.add_trace(
                go.Scatter(
                    x=[0, max_upload],
                    y=[0, max_upload],
                    mode="lines",
                    name="Ideal (y=x)",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        if self.download_history:
            download_estimates = [
                record["estimate"] for record in self.download_history
            ]
            download_actuals = [record["actual"] for record in self.download_history]
            download_time_ratios = [
                record["time_ratio"] for record in self.download_history
            ]

            # 创建hover text
            download_hover = [
                f"Estimate: {self.format_speed(est)}<br>"
                + f"Actual: {self.format_speed(act)}<br>"
                + f"Time Ratio: {tr:.2f}<br>"
                + f"Test #{i + 1}"
                for i, (est, act, tr) in enumerate(
                    zip(download_estimates, download_actuals, download_time_ratios)
                )
            ]

            # 添加下载散点图
            fig.add_trace(
                go.Scatter(
                    x=download_estimates,
                    y=download_actuals,
                    mode="markers",
                    name="Download Tests",
                    marker=dict(
                        size=10,
                        color=download_time_ratios,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Time Ratio", x=1.02, len=0.5),
                    ),
                    text=download_hover,
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            # 添加理想线 (y=x)
            max_download = max(max(download_estimates), max(download_actuals))
            fig.add_trace(
                go.Scatter(
                    x=[0, max_download],
                    y=[0, max_download],
                    mode="lines",
                    name="Ideal (y=x)",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # 更新布局
        fig.update_xaxes(
            title_text="Expected Speed (Bytes/s)", type="log", row=1, col=1
        )
        fig.update_xaxes(
            title_text="Expected Speed (Bytes/s)", type="log", row=1, col=2
        )
        fig.update_yaxes(title_text="Actual Speed (Bytes/s)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="Actual Speed (Bytes/s)", type="log", row=1, col=2)

        fig.update_layout(
            title_text="Bandwidth Probe Results Visualization",
            height=600,
            showlegend=True,
            hovermode="closest",
        )

        # 保存为HTML文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bandwidth_probe_results_{timestamp}.html"
        fig.write_html(filename)

        return filename


def handle_server(sock: TcpBrutalSocket):
    """服务器端处理逻辑"""
    sock.send_time()

    print("Server ready, waiting for commands...")

    while True:
        try:
            # 接收客户端命令
            command = sock.recv_json()
            action = command.get("action")
            # 仅在日志中打印已有字段，避免KeyError
            log_action = action if action is not None else "<missing>"
            log_size = command.get("size")
            log_rate = command.get("rate")
            print(
                f"Received command: {log_action}"
                + (f", size: {log_size} bytes" if log_size is not None else "")
                + (f", rate: {log_rate} B/s" if log_rate is not None else "")
            )

            if action is None:
                sock.send_json({"success": False, "error": "Missing action"})
                continue

            if action == "upload":
                size = command.get("size")
                if not isinstance(size, int) or size <= 0:
                    sock.send_json({
                        "success": False,
                        "error": "Invalid or missing size",
                    })
                    continue
                # 接收数据
                start_time = time.time()
                _ = sock.recv_data(size)
                actual_time = time.time() - start_time

                # 发送响应
                response = {
                    "success": True,
                    "actual_time": actual_time,
                    "bytes_transferred": size,
                }
                sock.send_json(response)
                print(f"  Upload completed in {actual_time:.2f}s")

            elif action == "download":
                size = command.get("size")
                rate = command.get("rate")
                if (
                    not isinstance(size, int)
                    or size <= 0
                    or not isinstance(rate, (int, float))
                    or rate <= 0
                ):
                    sock.send_json({
                        "success": False,
                        "error": "Invalid or missing size/rate",
                    })
                    continue
                # 设置发送速率
                sock.set_rate(rate)

                # 发送数据
                start_time = time.time()
                sock.send_data(size)
                actual_time = time.time() - start_time

                # 发送响应
                response = {
                    "success": True,
                    "actual_time": actual_time,
                    "bytes_transferred": size,
                }
                sock.send_json(response)
                print(f"  Download completed in {actual_time:.2f}s")

            elif action == "exit":
                print("Client requested exit")
                break

            else:
                response = {"success": False, "error": f"Unknown action: {action}"}
                sock.send_json(response)

        except Exception as e:
            print(f"Error handling command: {e}")
            break

    print("Server session ended")
    sock.close()


def handle_client(sock: TcpBrutalSocket):
    """客户端处理逻辑"""
    serverStamp = sock.recv_time()
    clientStamp = time.time()
    print(
        f"Server time: {datetime.fromtimestamp(serverStamp).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"Client time: {datetime.fromtimestamp(clientStamp).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Server-to-Client latency: {(clientStamp - serverStamp) * 1000:.2f}ms")
    print("If above metrics don't make sense, please check if the time is synced")
    time.sleep(1)

    # 开始带宽探测
    probe = BandwidthProbe(sock)
    probe.run_probe()

    # 生成可视化图表
    try:
        html_file = probe.generate_plots()
        print(f"\nVisualization saved to: {html_file}")
        print(f"Open the file in your browser to view the interactive plots.")
    except Exception as e:
        print(f"Failed to generate visualization: {e}")

    # 发送退出命令
    sock.send_json({"action": "exit"})
    sock.close()


def main():
    parser = argparse.ArgumentParser(description="TCP Brutal Socket Speed Probe")
    parser.add_argument(
        "--mode", choices=["server", "client"], default="client", required=True
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", default=30, type=int)
    parser.add_argument(
        "--duration",
        default=15,
        type=int,
        help="Test duration for each iteration (seconds)",
    )
    args = parser.parse_args()

    if args.mode == "server":
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", args.port))
        server_sock.listen(5)
        print(f"Server listening on 0.0.0.0:{args.port}")
        while True:
            conn, addr = server_sock.accept()
            print(f"Accepted connection from {addr}")
            conn.settimeout(args.timeout)
            brutal_sock = TcpBrutalSocket(conn)
            try:
                handle_server(brutal_sock)
            except Exception as e:
                print(f"Connection error: {e}")
            finally:
                print("Connection closed")

    elif args.mode == "client":
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((args.host, args.port))
        print(f"Connected to {args.host}:{args.port}")
        client_sock.settimeout(args.timeout)
        brutal_sock = TcpBrutalSocket(client_sock)
        try:
            handle_client(brutal_sock)
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            print("Client exiting")


if __name__ == "__main__":
    main()
