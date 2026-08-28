<template>
  <el-row :gutter="16">
    <el-col :span="6">
      <el-card shadow="never">
        <template #header><span>监控通道</span></template>
        <el-menu :default-active="String(currentId)" @select="onSelect">
          <el-menu-item v-for="d in devices" :key="d.id" :index="String(d.id)">
            <span>{{ d.device_name }}（{{ d.scene }}）</span>
            <el-tag size="small" :type="d.status === 'online' ? 'success' : 'info'" style="margin-left:8px;">{{ d.status }}</el-tag>
          </el-menu-item>
        </el-menu>
      </el-card>
    </el-col>
    <el-col :span="18">
      <el-card shadow="never">
        <template #header>
          <div class="card-header">
            <span>实时画面（模拟/测试流）</span>
            <el-button size="small" @click="loadDevices">刷新</el-button>
          </div>
        </template>
        <img v-if="currentId" :src="frameSrc" style="width:100%;" alt="实时画面" />
        <el-empty v-else description="请选择左侧监控通道" />
        <div style="margin-top:10px;">
          <el-descriptions v-if="meta" :column="4" size="small" border>
            <el-descriptions-item label="状态">{{ stateText }}</el-descriptions-item>
            <el-descriptions-item label="人数">{{ meta.person_count ?? '-' }}</el-descriptions-item>
            <el-descriptions-item label="跌倒">{{ meta.fall_detected ? '是' : '否' }}</el-descriptions-item>
            <el-descriptions-item label="跌倒概率">{{ probText(meta.fall_prob) }}</el-descriptions-item>
            <el-descriptions-item label="风险评分">{{ scoreText(meta.risk_score) }}</el-descriptions-item>
            <el-descriptions-item label="风险等级">{{ levelText }}</el-descriptions-item>
            <el-descriptions-item label="步态不稳">{{ probText(meta.gait_unsteadiness) }}</el-descriptions-item>
            <el-descriptions-item label="跌倒前兆">{{ probText(meta.nearfall_prob) }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import api from '../api'

const devices = ref([])
const currentId = ref(null)
const meta = ref(null)
const ts = ref(Date.now())
let frameTimer = null
let metaTimer = null

const frameSrc = computed(() => '/api/v1/streams/' + currentId.value + '/frame.jpg?t=' + ts.value)

const levelMap = { green: '正常', yellow: '注意', orange: '较高', red: '跌倒/严重' }
const stateText = computed(() => {
  if (!meta.value) return '-'
  if (meta.value.fall_detected) return '跌倒报警'
  return meta.value.source === 'rtsp' ? '直播中' : (meta.value.source === 'local_video' ? '播放中' : meta.value.state || '-')
})
const levelText = computed(() => levelMap[meta.value && meta.value.level] || '-')
function probText(v) { return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : '-' }
function scoreText(v) { return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : '-' }

async function loadDevices() {
  devices.value = await api.devices()
  if (!currentId.value && devices.value.length) currentId.value = devices.value[0].id
}
function onSelect(index) {
  currentId.value = Number(index)
  meta.value = null
}
async function refreshMeta() {
  if (!currentId.value) return
  try {
    const r = await fetch('/api/v1/streams/' + currentId.value + '/meta')
    const data = await r.json()
    meta.value = data.meta
  } catch (e) { /* ignore */ }
}
onMounted(() => {
  loadDevices()
  frameTimer = setInterval(() => { ts.value = Date.now() }, 120)
  metaTimer = setInterval(refreshMeta, 800)
  refreshMeta()
})
onBeforeUnmount(() => {
  if (frameTimer) clearInterval(frameTimer)
  if (metaTimer) clearInterval(metaTimer)
})
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>
