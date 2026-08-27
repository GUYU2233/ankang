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
        <img v-if="currentId" :src="frameSrc" style="width:100%; border-radius:6px;" alt="实时画面" />
        <el-empty v-else description="请选择左侧监控通道" />
        <div style="margin-top:10px;">
          <el-descriptions v-if="meta" :column="3" size="small" border>
            <el-descriptions-item label="状态">{{ meta.state || '-' }}</el-descriptions-item>
            <el-descriptions-item label="跌倒">{{ meta.fall_detected ? '是' : '否' }}</el-descriptions-item>
            <el-descriptions-item label="风险评分">{{ meta.risk_score }}</el-descriptions-item>
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
let timer = null

const frameSrc = computed(() => '/api/v1/streams/' + currentId.value + '/frame.jpg?t=' + ts.value)

async function loadDevices() {
  devices.value = await api.devices()
  if (!currentId.value && devices.value.length) currentId.value = devices.value[0].id
}
function onSelect(index) {
  currentId.value = Number(index)
  meta.value = null
  refresh()
}
async function refresh() {
  if (!currentId.value) return
  try {
    const r = await fetch('/api/v1/streams/' + currentId.value + '/meta')
    const data = await r.json()
    meta.value = data.meta
    ts.value = Date.now()
  } catch (e) { /* ignore */ }
}
onMounted(() => {
  loadDevices()
  timer = setInterval(refresh, 1000)
})
onBeforeUnmount(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>