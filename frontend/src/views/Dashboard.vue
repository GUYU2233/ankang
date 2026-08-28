<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="6" v-for="c in cards" :key="c.label">
        <el-card shadow="never">
          <div class="stat-value">{{ c.value }}</div>
          <div class="stat-label">{{ c.label }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>最新告警</span>
              <el-button size="small" @click="loadStats">刷新</el-button>
            </div>
          </template>
          <el-table :data="alerts" size="small" height="340">
            <el-table-column prop="level" label="级别" width="90">
              <template #default="s">
                <el-tag :type="levelType(s.row.level)" size="small">{{ levelText(s.row.level) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="title" label="内容" />
            <el-table-column prop="created_at" label="时间" width="180" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span>系统状态</span></template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="后端服务"><el-tag :type="online ? 'success' : 'danger'" size="small">{{ online ? '在线' : '离线' }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="WebSocket" ><el-tag :type="wsOpen ? 'success' : 'danger'" size="small">{{ wsOpen ? '已连接' : '未连接' }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="设备总数">{{ stats.total_devices }}</el-descriptions-item>
            <el-descriptions-item label="在线设备">{{ stats.online_devices }}</el-descriptions-item>
            <el-descriptions-item label="老人档案">{{ stats.total_residents }}</el-descriptions-item>
            <el-descriptions-item label="今日跌倒告警">{{ stats.today_falls }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="24">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>近7天风险趋势</span>
              <el-button size="small" @click="loadTrend">刷新</el-button>
            </div>
          </template>
          <div ref="trendEl" class="trend-box"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import * as echarts from 'echarts'
import { ElNotification } from 'element-plus'
import api from '../api'

const stats = ref({ total_devices: 0, online_devices: 0, total_residents: 0, today_alerts: 0, today_falls: 0, avg_risk_score: 0, latest_alerts: [] })
const online = ref(false)
const wsOpen = ref(false)
const trendEl = ref(null)
let ws = null
let timer = null
let trendChart = null

const cards = computed(() => [
  { label: '设备总数', value: stats.value.total_devices },
  { label: '在线设备', value: stats.value.online_devices },
  { label: '今日告警', value: stats.value.today_alerts },
  { label: '平均风险评分', value: stats.value.avg_risk_score }
])
const alerts = computed(() => stats.value.latest_alerts || [])

function levelType(level) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[level] || 'info' }
function levelText(level) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[level] || level }

async function loadStats() {
  try {
    const data = await api.dashboard()
    stats.value = data
    online.value = true
  } catch (e) {
    online.value = false
  }
}

async function loadTrend() {
  try {
    const data = await api.riskTrend(7)
    const trend = Array.isArray(data) ? data : (data.trend || [])
    if (!trendEl.value) return
    if (!trendChart) trendChart = echarts.init(trendEl.value)
    trendChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['平均风险分', '峰值'] },
      grid: { left: 40, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: trend.map(p => p.date) },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: [
        { name: '平均风险分', type: 'line', smooth: true, areaStyle: {}, data: trend.map(p => p.avg_score) },
        { name: '峰值', type: 'line', smooth: true, lineStyle: { type: 'dashed' }, data: trend.map(p => p.max_score) }
      ]
    })
  } catch (e) { /* 趋势加载失败不打断看板 */ }
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(proto + '://' + location.host + '/ws/alerts')
  ws.onopen = () => { wsOpen.value = true }
  ws.onclose = () => { wsOpen.value = false }
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'alert') {
        const d = msg.data || {}
        const name = d.resident_name ? d.resident_name + ' · ' : ''
        const msgText = name + (d.title || '') + ' [' + (d.scene || '') + ']' + (d.guardian_phone ? '（家属: ' + d.guardian_phone + '）' : '')
        ElNotification({ title: '收到预警', message: msgText, type: d.level === 'red' ? 'error' : 'warning', duration: 6000 })
        loadStats()
        loadTrend()
      }
    } catch (e) { /* ignore */ }
  }
}

function onResize() {
  if (trendChart) trendChart.resize()
}

onMounted(() => {
  loadStats()
  loadTrend()
  connectWs()
  timer = setInterval(loadStats, 5000)
  window.addEventListener('resize', onResize)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  if (ws) ws.close()
  window.removeEventListener('resize', onResize)
  if (trendChart) { trendChart.dispose(); trendChart = null }
})
</script>

<style scoped>
.stat-value { font-size: 30px; font-weight: 700; color: #303133; }
.stat-label { color: #909399; margin-top: 6px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.trend-box { height: 300px; width: 100%; }
</style>